import moe_from_dense.parser as args_parser
from moe_from_dense.make_moe import create_moe_model


def main():
    parser = args_parser.get_parser()
    args = parser.parse_args()

    create_moe_model(args)

if __name__ == '__main__':
    main()


