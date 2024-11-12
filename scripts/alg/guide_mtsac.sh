env=$1
map=$2
replay_buffer_capacity=$3
replay_buffer_batch_size=$4
num_train_steps=$5

env_name="$env-$map"

PYTHONPATH=. python3 -u main.py \
setup.alg=guide_mtsac \
metrics=mtrl_guide \
env=$env_name \
agent=guide_sac \
experiment.name=$env \
experiment.num_train_steps=$num_train_steps \
experiment.use_guide=True \
replay_buffer.capacity=$replay_buffer_capacity \
replay_buffer.batch_size=$replay_buffer_batch_size \
agent.encoder.type_to_select=identity \
agent.multitask.should_use_disentangled_alpha=True \
agent.multitask.should_use_task_encoder=False \
agent.multitask.should_use_task_onehot=True \
agent.multitask.should_use_multi_head_policy=False \
agent.multitask.actor_cfg.should_condition_model_on_task_info=False \
agent.multitask.actor_cfg.should_condition_encoder_on_task_info=False \
agent.multitask.actor_cfg.should_concatenate_task_info_with_encoder=False \
agent.guide_encoder.type_to_select=identity \
agent.builder.guide_hindsight=True
