
declare -a arr=("TB-DENSE MATRES TIMELINE")

model_name="roberta-base"
# model_name_large="roberta-large"

for dataset in "${arr[@]}"
do

    python3 main.py --config_file=configs/word_conf_linear_dual.json --device=cuda:0 --dataset="${dataset}" --model_name="${model_name}"& 
    python3 main.py --config_file=configs/word_conf_linear_dual_cls_as_context_robLarge.json --device=cuda:1 --dataset="${dataset}"   --model_name="${model_name_large}"

    # python3 main.py --config_file=configs/word_conf_linear_llama2_7b_frozen.json --device=cuda --dataset="${dataset}"
    # python3 main.py --config_file=configs/word_conf_linear_llama2_13b_frozen.json --device=cuda --dataset="${dataset}"
    # python3 main.py --config_file=configs/word_conf_linear_llama2_70b_frozen.json --device=cuda --dataset="${dataset}"


done

