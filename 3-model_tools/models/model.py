from tqdm import tqdm
from utils import *
from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer
import torch.optim as optim
from lr_scheduler import get_linear_schedule_with_warmup

class BasicModel():
    def __init__(self, args):
        self.model, self.tokenizer = self._get_model_tokenizer(args)
        self.args = args
        self.t_total = len(
            args.train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        bart_param_optimizer = list(self.model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bart_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay, 'lr': args.learning_rate},
            {'params': [p for n, p in bart_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
             'lr': args.learning_rate}
        ]
        args.warmup_steps = int(self.t_total * args.warmup_proportion)
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps,
                                                         num_training_steps=self.t_total)

    def _get_model_tokenizer(self, args):
        config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path).to(args.device)
        return model, tokenizer

    def train_model(self):
        logging.info("Num examples = %d", len(self.args.train_set))
        logging.info("Num Epochs = %d", self.args.num_train_epochs)
        logging.info("Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logging.info("Total optimization steps = %d", self.t_total)
        logging.info("Begin training ")

        global_step = 0
        steps_trained_in_current_epoch = 0
        logging.info('Checking...')
        tr_loss, logging_loss = 0.0, 0.0

        acc_max = 0
        device = self.args.device
        model = self.model
        args = self.args
        optimizer = self.optimizer
        scheduler = self.scheduler
        tokenizer = self.tokenizer
        model.zero_grad()
        for epoch in range(int(self.args.num_train_epochs)):
            pbar = ProgressBar(n_total=len(args.train_loader), desc='Training')
            for step, batch in enumerate(args.train_loader):
                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                model.train()
                batch = tuple(t.to(device) for t in batch)
                pad_token_id = self.tokenizer.pad_token_id
                source_ids, source_mask, y = batch[0], batch[1], batch[2]
                y_ids = y[:, :-1].contiguous()
                lm_labels = y[:, 1:].clone()
                lm_labels[y[:, 1:] == pad_token_id] = -100

                inputs = {
                    "input_ids": source_ids.to(device),
                    "attention_mask": source_mask.to(device),
                    "decoder_input_ids": y_ids.to(device),
                    "labels": lm_labels.to(device),
                }

                outputs = model(**inputs)
                loss = outputs[0]
                loss.backward()
                pbar(step, {'loss': loss.item()})
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            # Save model checkpoint
            if epoch / args.valid_epoch == 0:
                acc = validate(model, args.valid_loader, device, tokenizer)
                if acc > acc_max:
                    acc_max = acc
                    # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    output_dir = args.output_dir
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed.txt/parallel training
                    model_to_save.save_pretrained(os.path.join(output_dir, 'model'))
                    tokenizer.save_pretrained(os.path.join(output_dir, 'tokenizer'))
                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logging.info("Saving model checkpoint to %s", output_dir)

                    # tokenizer.save_vocabulary(output_dir)
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logging.info("Saving optimizer and scheduler states to %s", output_dir)
                    logging.info("\n")
            if 'cuda' in str(device):
                torch.cuda.empty_cache()
        return global_step, tr_loss / global_step


def validate(model, data, device, tokenizer):
    f = open("./check.txt", "w")
    model.eval()
    count, correct = 0, 0
    with torch.no_grad():
        all_outputs = []
        all_gts = []
        all_questions = []
        for batch in tqdm(data, total=len(data)):
            source_ids, source_mask, target_ids = [x.to(device) for x in batch]
            source_ids = source_ids.cuda()
            outputs = model.generate(
                input_ids=source_ids,
                max_length=500,
            )
            # src
            all_questions.extend(source_ids.cpu().numpy())
            # output
            all_outputs.extend(outputs.cpu().numpy())
            # target
            all_gts.extend(target_ids.cpu().numpy())

        outputs = [tokenizer.decode(output_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for
                   output_id in all_outputs]
        gts = [tokenizer.decode(gt, skip_special_tokens=True, clean_up_tokenization_spaces=True) for gt in all_gts]

        for gt, output in tqdm(zip(gts, outputs)):
            output = output.strip()
            if (gt == output):
                correct += 1
                f.write("tgt:" + gt + ",predict:" + output + ",right" + "\n")
            else:
                f.write("tgt:" + gt + ",predict:" + output + ",wrong" + "\n")
                print("********", gt, "------", output)
            count += 1
        acc = correct / count
        logging.info('acc: {}'.format(acc))

        return acc

def test(args):
    logging.info("loading model.........")
    config_class, model_class, tokenizer_class = (BartConfig, BartForConditionalGeneration, BartTokenizer)
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.load_state_dict(torch.load('./output/checkpoint-29688/model/pytorch_model.bin'))
    model=model.to(args.device)
    logging.info('success init model')
    val_set = args.valid_set
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False,num_workers=10)
    validate(model, val_loader, args.device, tokenizer)
