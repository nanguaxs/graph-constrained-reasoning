import json
import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any
import openai


class KnowledgeGraph:
    def __init__(self, triples: List[List[str]]):
        self.triples = triples
        self.forward_index = defaultdict(list)
        self.backward_index = defaultdict(list)
        self.entities = set()
        self.relations = set()
        self._build_index()
    
    def _build_index(self):
        for head, relation, tail in self.triples:
            self.forward_index[head].append((relation, tail))
            self.backward_index[tail].append((relation, head))
            self.entities.add(head)
            self.entities.add(tail)
            self.relations.add(relation)
    
    def get_neighbors(self, entity: str, direction='forward'):
        if direction == 'forward':
            return self.forward_index.get(entity, [])
        else:
            return self.backward_index.get(entity, [])
    
    def get_all_entities(self):
        return list(self.entities)
    
    def get_triples(self):
        return self.triples


class PathSampler:
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def sample_one_hop_paths(self, num_samples: int) -> List[Dict]:
        paths = []
        entities = self.kg.get_all_entities()
        
        attempts = 0
        max_attempts = num_samples * 50
        
        while len(paths) < num_samples and attempts < max_attempts:
            attempts += 1
            entity = random.choice(entities)
            neighbors = self.kg.get_neighbors(entity, 'forward')
            
            if not neighbors:
                continue
            
            relation, tail = random.choice(neighbors)
            
            path = {
                'type': '1-hop',
                'start_entity': entity,
                'end_entity': tail,
                'path': [(entity, relation, tail)],
                'relations': [relation]
            }
            # 打印采样的1跳路径
            print(f"  采样到1跳路径: {entity} -[{relation}]-> {tail}")
            paths.append(path)
        
        return paths
    
    def sample_two_hop_paths(self, num_samples: int) -> List[Dict]:
        paths = []
        entities = self.kg.get_all_entities()
        
        attempts = 0
        max_attempts = num_samples * 100
        
        while len(paths) < num_samples and attempts < max_attempts:
            attempts += 1
            start_entity = random.choice(entities)
            
            # First hop
            neighbors_1 = self.kg.get_neighbors(start_entity, 'forward')
            if not neighbors_1:
                continue
            
            relation_1, middle_entity = random.choice(neighbors_1)
            
            # Second hop
            neighbors_2 = self.kg.get_neighbors(middle_entity, 'forward')
            if not neighbors_2:
                continue
            
            relation_2, end_entity = random.choice(neighbors_2)
            
            # Avoid trivial paths
            if start_entity == end_entity:
                continue
            
            path = {
                'type': '2-hop',
                'start_entity': start_entity,
                'middle_entity': middle_entity,
                'end_entity': end_entity,
                'path': [
                    (start_entity, relation_1, middle_entity),
                    (middle_entity, relation_2, end_entity)
                ],
                'relations': [relation_1, relation_2]
            }
            # 打印采样的2跳路径
            print(f"  采样到2跳路径: {start_entity} -[{relation_1}]-> {middle_entity} -[{relation_2}]-> {end_entity}")
            paths.append(path)
        
        return paths


class LLMQuestionGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
    
    def generate_one_hop_question(self, path: Dict) -> Dict:
        head = path['start_entity']
        relation = path['relations'][0]
        tail = path['end_entity']
        
        prompt = f"""给定知识图谱三元组: [{head}, {relation}, {tail}]

任务: 生成一个自然语言问题,满足以下要求:
1. 问题中包含实体"{head}"
2. 答案是实体"{tail}"
3. 问题表达自然流畅,符合中文习惯
4. 问题需要通过关系"{relation}"进行推理

示例:
三元组: [三全食品股份有限公司, 董事长, 陈南]
问题: 三全食品股份有限公司的董事长是谁?

三元组: [农林牧渔业, 包括, 农业]
问题: 农林牧渔业包括哪些行业?

请只输出问题,不要输出其他内容:"""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        question = response.choices[0].message.content.strip()
        
        # 打印生成的1跳问答对
        print(f"    生成问答对: Q: {question} A: {tail}")
        
        return {
            'question': question,
            'answer': [tail],
            'q_entity': [head],
            'a_entity': [tail]
        }
    
    def generate_two_hop_question(self, path: Dict) -> Dict:
        start = path['start_entity']
        middle = path['middle_entity']
        end = path['end_entity']
        relation_1 = path['relations'][0]
        relation_2 = path['relations'][1]
        
        prompt = f"""给定知识图谱推理路径:
- 起点实体: {start}
- 第1跳: {start} -[{relation_1}]-> {middle}
- 第2跳: {middle} -[{relation_2}]-> {end}

任务: 生成一个需要两步推理的自然语言问题:
1. 问题中包含起点实体"{start}"
2. 答案是目标实体"{end}"
3. 回答问题需要经过中间实体"{middle}"
4. 问题表达自然,不要明显暴露推理步骤

示例:
路径: 农副食品加工 -[上市公司]-> 三全食品股份有限公司 -[董事长]-> 陈南
问题: 农副食品加工行业的上市公司三全食品的董事长是谁?

路径: 金融业 -[属于]-> 保险业 -[上市公司]-> 某保险公司
问题: 金融业下属的保险业有哪些上市公司?

请只输出问题,不要输出其他内容:"""

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        question = response.choices[0].message.content.strip()
        
        # 打印生成的2跳问答对
        print(f"    生成问答对: Q: {question} A: {end}")
        
        return {
            'question': question,
            'answer': [end],
            'q_entity': [start],
            'a_entity': [end]
        }
    
    def generate_question(self, path: Dict) -> Dict:
        if path['type'] == '1-hop':
            return self.generate_one_hop_question(path)
        else:
            return self.generate_two_hop_question(path)


class KGQADatasetGenerator:
    def __init__(self, kg_file: str, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None):
        with open(kg_file, 'r', encoding='utf-8') as f:
            triples = json.load(f)
        
        self.kg = KnowledgeGraph(triples)
        self.sampler = PathSampler(self.kg)
        self.llm_generator = LLMQuestionGenerator(api_key, model, base_url)
        self.graph_triples = triples
    
    def generate_dataset(self, num_samples: int, one_hop_ratio: float = 0.5) -> List[Dict]:
        num_one_hop = int(num_samples * one_hop_ratio)
        num_two_hop = num_samples - num_one_hop
        
        print(f"开始采样 {num_one_hop} 个1跳路径...")
        one_hop_paths = self.sampler.sample_one_hop_paths(num_one_hop)
        print(f"成功采样 {len(one_hop_paths)} 个1跳路径\n")
        
        print(f"开始采样 {num_two_hop} 个2跳路径...")
        two_hop_paths = self.sampler.sample_two_hop_paths(num_two_hop)
        print(f"成功采样 {len(two_hop_paths)} 个2跳路径\n")
        
        all_paths = one_hop_paths + two_hop_paths
        random.shuffle(all_paths)
        
        dataset = []
        
        print(f"\n开始生成问答对...")
        for idx, path in enumerate(all_paths):
            try:
                print(f"正在生成第 {idx+1}/{len(all_paths)} 个问答对...")
                
                # 打印当前处理的路径
                if path['type'] == '1-hop':
                    print(f"    处理1跳路径: {path['start_entity']} -[{path['relations'][0]}]-> {path['end_entity']}")
                else:
                    print(f"    处理2跳路径: {path['start_entity']} -[{path['relations'][0]}]-> {path['middle_entity']} -[{path['relations'][1]}]-> {path['end_entity']}")
                    
                qa = self.llm_generator.generate_question(path)
                
                qa_item = {
                    'id': f"kg_qa_{idx+1}",
                    'question': qa['question'],
                    'answer': qa['answer'],
                    'q_entity': qa['q_entity'],
                    'a_entity': qa['a_entity'],
                    'graph': self.graph_triples,
                    'reasoning_path': path['path']
                }
                
                dataset.append(qa_item)
                print(f"    问答对生成成功！")
                
            except Exception as e:
                print(f"生成第 {idx+1} 个问答对时出错: {e}")
                continue
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\n数据集已保存到: {output_file}")


def main():
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    generator = KGQADatasetGenerator(
        kg_file=config['kg_file'],
        api_key=config['api_key'],
        model=config['model'],
        base_url=config.get('base_url')
    )
    
    dataset = generator.generate_dataset(
        num_samples=config['num_samples'],
        one_hop_ratio=config['one_hop_ratio']
    )
    
    generator.save_dataset(dataset, config['output_file'])
    
    print(f"\n共生成 {len(dataset)} 个问答对")


if __name__ == "__main__":
    main()
