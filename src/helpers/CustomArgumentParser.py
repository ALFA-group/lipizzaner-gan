import argparse
import sys


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()

        sys.stderr.write("\nError: %s\n" % message)
        sys.exit(2)
