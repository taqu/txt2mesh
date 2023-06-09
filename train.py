import torch
import datasets
import sys
import os
import subprocess

def load_count(filename):
    count = 0
    try:
        with open(filename, mode='r') as f:
            for l in f.readlines():
                try:
                    count = int(l)
                    break
                except:
                    continue
    except:
        pass
    finally:
        return count

def save_count(filename, count):
    try:
        with open(filename, mode='w') as f:
            f.write(str(count))
    except:
        pass
    finally:
        return

prompts = []
dataset = datasets.load_dataset("FredZhang7/stable-diffusion-prompts-2.47M")
prompts = prompts + dataset['train']['text']

dataset = datasets.load_dataset('Gustavosta/Stable-Diffusion-Prompts')
prompts = prompts + dataset['train']['Prompt']

#dataset = datasets.load_dataset("FredZhang7/anime-prompts-180K")
#prompts = prompts + dataset['train']['safebooru_clean']
#prompts = prompts + dataset['train']['danbooru_clean']
#prompts = prompts + dataset['train']['danbooru_raw']                         

dataset = None

train_count = load_count('train_count.txt')
print(min(len(prompts), train_count))
max_train_count = max(len(prompts), train_count)

conf_file = os.path.join(os.getcwd(), 'threestudio/configs/prolificdreamer.yaml')
print(conf_file)
os.chdir('./threestudio')
torch.set_float32_matmul_precision('medium')
#base_config = ['launch.py','--config', 'configs/prolificdreamer-refine.yaml', '--train', '--gpu', '0', 'data.width=512', 'data.height=512']
base_config = ['python3', 'launch.py','--config', conf_file, '--train', '--gpu', '0', 'name=\"sheep\"','tag=\"sheep\"','data.width=1', 'data.height=1']
#extra_condig = ['trainer.max_steps=10000', 'system.guidance.token_merging=true', 'system.guidance.enable_attention_slicing=true']
extra_condig = ['trainer.max_steps=1', 'trainer.max_epochs=1', 'trainer.fast_dev_run=True', 'data.batch_size=1', 'system.guidance.token_merging=true', 'system.guidance.enable_attention_slicing=true']
for i in range(0,1):
    config = base_config + ['system.prompt_processor.prompt=' + prompts[i]]
    print(config)
    subprocess.run(config, capture_output=False)
    train_count += 1
    save_count('train_count.txt', train_count)
os.chdir('..')

