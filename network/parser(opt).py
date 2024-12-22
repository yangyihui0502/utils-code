# 用parser获取opt，获取后可以用.来引用
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', help='cuda device')
    parser.add_argument('--data_root_folder', default="data", help='')
    parser.add_argument('--window', type=int, default=120, help='horizon')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')
    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    # For testing sampled results 
    parser.add_argument("--test_sample_res", action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()