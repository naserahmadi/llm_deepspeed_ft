from datasets import load_dataset
import json


qa_dataset = load_dataset('lucadiliello/triviaqa')
INST = """Answer the following question based on the context. Your answer should be based on the provided context only."""


def create_inst(sample):
    inpt = f"""\n### Question:\n{sample['question']}\n\n### Context:\n{sample['context']}"""
    outpt = f"""### Answer:\n{sample['answers'][0]}"""
    return {'instruction': INST, 'input': inpt, 'output': outpt}


train_qa, val_qa, test_qa = [],[],[]

for i in range(0, len(qa_dataset['train'])):
    sample = create_inst(qa_dataset['train'][i])
    train_qa.append(sample)

for i in range(0, len(qa_dataset['validation'])):
    sample = create_inst(qa_dataset['validation'][i])
    val_qa.append(sample)


with open('dataset/train.json', 'w') as f:
    json.dump(train_qa,f)

with open('dataset/val.json', 'w') as f:
    json.dump(val_qa,f)

