import argparse


def parameter_parser():
    # 实验参数
    parser = argparse.ArgumentParser(description="siamese network")
    # 数据库
    parser.add_argument("-D", '--dataset', type=str, default='train_data/reentrancy_1671.txt')
    parser.add_argument('--vector_dim', type=int, default=300,)
    parser.add_argument("--batch_size", type=int, default=128,)
    parser.add_argument("--epochs", type=int, default=30,)
    parser.add_argument("--lr", type=float, default=-0.003,)
    parser.add_argument("--dropout", type=float, default=0.2,)
    parser.add_argument("--threshold", type=float, default=0.5,)
    parser.add_argument("--time_steps", type=int, default=28,)
    parser.add_argument("--input_size", type=int, default=28,)
    parser.add_argument("--batch_index", type=int, default=0,)
    parser.add_argument("--output_size", type=int, default=10,)
    parser.add_argument("--cell_size", type=int, default=300,)
    parser.add_argument("--num_classes", type=int, default=2,)
    return parser.parse_args()