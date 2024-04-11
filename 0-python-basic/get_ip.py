def get_node_from_ip(ifname='eth0'):
    result = subprocess.run(['ifconfig', ifname], stdout=subprocess.PIPE)
    output = result.stdout.decode()
    ip_address = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', output)
    if not ip_address:
        return 'Unknown node'
    ip_address = ip_address.group(1)
    ip_to_node = {
        '192.69.107.248': '0',
        '192.4.183.6': '1',
        '192.216.99.31': '2'
    }
    node_identifier = ip_to_node.get(ip_address, 'Unknown node')
    return node_identifier