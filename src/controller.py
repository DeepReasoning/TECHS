import os



def run_TLoGN():
    # max_nodes_list = [20, 40, 60, 100]  # 40
    # sample_nodes_list = [15, 50, 100, 600]  # 15
    max_nodes_list = [120]  # 40
    sample_nodes_list = [600]  # 15
    sample_method = 3  # 1 2 3所有 前向 后向
    sample_ratio = 0.5
    logic_ratio = 0.8
    score_method = 'att'  # emd att both
    loss = 'bce'  # bce max_min focal
    use_gcn = 1  # 不使用GCN
    logic_layers = [3]
    dataset = 'icews14'  # icews0515 WIKI YAGO
    time_score = 0  # 计算得分时是否使用time

    gpu = 2

    for max_nodes in max_nodes_list:
        for sample_nodes in sample_nodes_list:
            for logic_layer in logic_layers:
                temp_cmd = f'python run_model.py --max_nodes {max_nodes} --sample_nodes {sample_nodes}' \
                           f' --sample_method {sample_method} --sample_ratio {sample_ratio}' \
                           f' --score_method {score_method} --loss {loss} --use_gcn {use_gcn}' \
                           f' --time_score {time_score} --gpu {gpu} --logic_ratio {logic_ratio} ' \
                           f'--logic_layer {logic_layer} --dataset {dataset}'
                os.system(temp_cmd)

run_TLoGN()

# def view_results():




def run_gcn():
    # gcn_layers_list = [1,2]
    # dataset_list = ['icews14', 'icews18', 'icews0515']

    gcn_layers_list = [1]
    dataset_list = ['icews14']

    gpu = 6

    for gcn_layer in gcn_layers_list:
        for dataset in dataset_list:
            temp_cmd = f'python run_model.py --gpu {gpu} --gcn_layer {gcn_layer} --dataset {dataset}'
            os.system(temp_cmd)

# run_gcn()

