import src.parser as args_parser
from src.make_moe import create_moe_model


def main():
    parser = args_parser.get_parser()
    args = parser.parse_args()

    create_moe_model(args)

if __name__ == '__main__':
    main()


