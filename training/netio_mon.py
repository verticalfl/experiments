import atexit
import subprocess
import sys

# Create unique chain names for each direction
outbound_chain = "TRAFFIC_MONITOR_OUT"
inbound_chain = "TRAFFIC_MONITOR_IN"


def setup_traffic_monitoring(port=None, ip=None):
    """
    Sets up iptables rules to monitor both inbound and outbound traffic.

    Args:
        port (int, optional): Specific port to monitor
        ip (str, optional): Specific IP to monitor
    """

    def cleanup():
        # Remove from OUTPUT and INPUT chains
        subprocess.run(
            ["sudo", "iptables", "-D", "OUTPUT", "-j", outbound_chain], check=False
        )
        subprocess.run(
            ["sudo", "iptables", "-D", "INPUT", "-j", inbound_chain], check=False
        )
        # Flush and delete our chains
        for chain in [outbound_chain, inbound_chain]:
            subprocess.run(["sudo", "iptables", "-F", chain], check=False)
            subprocess.run(["sudo", "iptables", "-X", chain], check=False)

    # If the chains already exist from a previous run, clean
    cleanup()

    try:
        # Create new chains
        for chain in [outbound_chain, inbound_chain]:
            subprocess.run(["sudo", "iptables", "-N", chain], check=True)

        # Create base rule for inbound chain
        inbound_rule = []
        if ip:
            inbound_rule.extend(["-s", ip])
        if port:
            inbound_rule.extend(["-p", "tcp", "--dport", str(port)])

        # Create base rule for outbound chain
        outbound_rule = []
        if ip:
            outbound_rule.extend(["-d", ip])
        if port:
            outbound_rule.extend(["-p", "tcp", "--dport", str(port)])

        # Add rules to each chain
        subprocess.run(
            ["sudo", "iptables", "-A", inbound_chain] + inbound_rule, check=True
        )
        subprocess.run(
            ["sudo", "iptables", "-A", outbound_chain] + outbound_rule, check=True
        )

        # Link to OUTPUT and INPUT chains
        subprocess.run(
            ["sudo", "iptables", "-I", "OUTPUT", "1", "-j", outbound_chain], check=True
        )
        subprocess.run(
            ["sudo", "iptables", "-I", "INPUT", "1", "-j", inbound_chain], check=True
        )

        # Register cleanup function
        atexit.register(cleanup)

    except subprocess.CalledProcessError as e:
        print(f"Error setting up iptables: {e}", file=sys.stderr)
        sys.exit(1)


def get_byte_count(direction="both"):
    """
    Returns the number of bytes counted by the monitoring chains.

    Args:
        direction (str): "in", "out", or "both"

    Returns:
        int or tuple: Byte count(s) depending on direction specified
    """
    try:

        def get_chain_bytes(chain):
            result = subprocess.run(
                ["sudo", "iptables", "-L", chain, "-v", "-n", "-x"],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = result.stdout.split("\n")
            if len(lines) >= 3:
                fields = lines[2].split()
                if len(fields) >= 2:
                    return int(fields[1])
            return 0

        if direction == "out":
            return get_chain_bytes(outbound_chain)
        elif direction == "in":
            return get_chain_bytes(inbound_chain)
        else:  # both
            return (get_chain_bytes(inbound_chain), get_chain_bytes(outbound_chain))

    except subprocess.CalledProcessError as e:
        print(f"Error getting byte count: {e}", file=sys.stderr)
        return 0 if direction != "both" else (0, 0)