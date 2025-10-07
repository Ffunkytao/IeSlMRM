

from eval_src.toolkit_for_MATH.latex_answer_check import latex_answer_check as latex_equiv

import os, json, re
import sqlite3
from typing import List, Dict, Tuple
from collections import defaultdict
import random
from fuzzywuzzy import fuzz, process
import pymysql
from tqdm import tqdm  # 导入 tqdm
import json

evaluatefilequeryfile='right_sql.json'


class Evaluator:
    def __init__(self) -> None:
        self.answer_marker = "answer is"

    def _is_number(self, s) -> Tuple[bool, str]:
        try:
            res = float(s)
            return True, str(res)
        except:
            pass
        try:
            import unicodedata

            res = unicodedata.numeric(s)
            return True, str(res)
        except:
            pass
        return False, None

    def validate_completion(self, completion: str) -> bool:
        if self.answer_marker.lower() in completion.lower():
            return True

        return False

    def isolate_answer(self, text: str):
        if text is None:
            return None

        assert isinstance(text, str)
        text = text.lower()
        split_ans = text.split(self.answer_marker.lower())
        if len(split_ans) > 1:
            ans = split_ans[-1].replace(":", "").strip()
            extract_ans_temp = ans.split(".\n")[0].strip()
            if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == ".":
                extract_ans = extract_ans_temp[0:-1]
            else:
                extract_ans = extract_ans_temp
            extract_ans = extract_ans.strip().strip("\n")
            return extract_ans
        else:
            return text

    def find_most_confident_answer(self, completions: List[str], prior_weights: List[float] = None):
        """Returns the most confident answer, its completion, its id in the input list, and its confidence."""
        if completions is None or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        answer2ids = defaultdict(list)
        for id, c in enumerate(completions):
            try:
                model_answer = self.extract_answer_from_model_completion(c)
                has_existed = False
                for existing_answer in answer2completions.keys():
                    if self.check_answers_equiv(model_answer, existing_answer):
                        assert not has_existed
                        has_existed = True
                        answer2completions[existing_answer].append(c)
                        answer2ids[existing_answer].append(id)
                if not has_existed:
                    answer2completions[model_answer].append(c)
                    answer2ids[model_answer].append(id)
            except:
                pass

        assert len(answer2completions.keys()) > 0, "There are no valid completions."
        if prior_weights is not None:
            assert len(completions) == len(prior_weights)
            completion2count = {}
            for answer, answer_completions in answer2completions.items():
                count = len(answer_completions)
                for answer_completion in answer_completions:
                    completion2count[answer_completion] = count

            completion2score = {}
            for id, (completion, count) in enumerate(completion2count.items()):
                prior_weight = prior_weights[id]
                score = prior_weight * (count / len(completions))
                completion2score[completion] = score

            most_confident_completion = max(completion2score.keys(), key=lambda x: completion2score[x])

            return (
                self.extract_answer_from_model_completion(most_confident_completion),
                most_confident_completion,
                completions.index(most_confident_completion),
                completion2score[most_confident_completion],
            )
        else:
            most_confident_answer = max(answer2completions.keys(), key=lambda x: len(answer2completions[x]))
            assert (
                len(answer2completions[most_confident_answer]) > 0
            ), "There are no completions for the most confident answer."
            confidence = len(answer2completions[most_confident_answer]) / len(completions)
            assert confidence > 0
            return (
                most_confident_answer,
                answer2completions[most_confident_answer][0],
                answer2ids[most_confident_answer][0],
                confidence,
            )

    def stochastic_select_answer(self, completion2score, answer2completions, completions):
        answer2score = {}
        answer_counts = {}
        for completion, score in completion2score.items():
            answer = self.extract_answer_from_model_completion(completion)
            if answer in answer2score:
                answer2score[answer] += score
                answer_counts[answer] += 1
            else:
                answer2score[answer] = score
                answer_counts[answer] = 1

        for answer in answer2score:
            answer2score[answer] /= answer_counts[answer]

        top_answers = sorted(answer2score.items(), key=lambda x: x[1], reverse=True)[:1]
        answers, scores = zip(*top_answers)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            selected_answer = random.choices(answers, weights=probabilities, k=1)[0]
        except:
            selected_answer = random.choices(answers, k=1)[0]

        most_confident_completion = answer2completions[selected_answer][0]
        completion_index = completions.index(most_confident_completion)
        confidence = answer2score[selected_answer]

        return selected_answer, most_confident_completion, completion_index, confidence

    def stochastic_calculate_completion_scores(self, prior_weights, answer2completions):
        completion2count = {}
        for answer, comps in answer2completions.items():
            count = len(comps)
            for comp in comps:
                completion2count[comp] = count

        completion2score = {}
        for idx, comp in enumerate(completion2count.keys()):
            weight = prior_weights[idx] if prior_weights is not None else 1
            score = weight * completion2count[comp]
            completion2score[comp] = score
        return completion2score

    def stochastic_select_response(self, completion2score, completions):
        sorted_completions = sorted(completion2score.items(), key=lambda x: x[1], reverse=True)[:1]
        completions, scores = zip(*sorted_completions)
        total_score = sum(scores)
        try:
            probabilities = [score / total_score for score in scores]
            sampled_completion = random.choices(completions, weights=probabilities, k=1)[0]
        except:
            sampled_completion = random.choices(completions, k=1)[0]
        confidence = completion2score[sampled_completion]
        most_confident_answer = self.extract_answer_from_model_completion(sampled_completion)
        id_of_most_confident = completions.index(sampled_completion)
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def stochastic_find_most_confident_answer(
        self,
        completions: List[str],
        prior_weights: List[float] = None,
    ):

        if not completions or len(completions) == 0:
            return None, None, None, None

        answer2completions = defaultdict(list)
        for idx, comp in enumerate(completions):
            try:
                answer = self.extract_answer_from_model_completion(comp)
                answer2completions[answer].append(comp)
            except:
                continue

        if not answer2completions:
            return None, None, None, None

        completion2score = self.stochastic_calculate_completion_scores(prior_weights, answer2completions)

        most_confident_answer, sampled_completion, id_of_most_confident, confidence = self.stochastic_select_response(
            completion2score, completions
        )
        return most_confident_answer, sampled_completion, id_of_most_confident, confidence

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        raise NotImplementedError

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        raise NotImplementedError

    def extract_answer_from_model_completion(self, completion: str) -> str:
        raise NotImplementedError


class GSM8KEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__()

    def check_answers_equiv(self, answer_a: str, answer_b: str):
        """Judge whether two answers are equivalent."""
        is_number_a, number_a = self._is_number(answer_a)
        is_number_b, number_b = self._is_number(answer_b)
        if is_number_a and is_number_b:
            correct = number_a == number_b
        else:
            correct = False

        return correct

    def extract_answer_from_gold_solution(self, solution: str | float):
        """Extract the answer from the gold solution."""
        if isinstance(solution, float):
            return str(solution)
        return solution.split("#### ")[-1].strip()

    def extract_answer_from_model_completion(self, completion: str):
        """Extract the answer from the model completion."""
        if completion is None:
            return None

        assert isinstance(completion, str)

        preds = completion
        preds = preds.split(self.answer_marker)
        answer_flag = True if len(preds) > 1 else False
        if answer_flag:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return None
        else:
            if answer_flag:
                pred = pred[0]
            else:
                pred = pred[-1]

        if pred != "" and pred[-1] == ".":
            pred = pred[:-1]

        pred = pred.replace(",", "").replace("\n", "")
        is_number, pred = self._is_number(pred)
        if is_number:
            return pred
        else:
            return None


GSM8KHARDEvaluator = GSM8KEvaluator
MULTIARITHEvaluator = GSM8KEvaluator



class LOGICCATEvaluator(Evaluator):
    """
    专用于 LOGICCAT 数据集的 Evaluator，进行 SQL 精确匹配评估。
    """
    def __init__(self) -> None:
        super().__init__()
        # 提取答案时，查找 'answer is' 标记
        self.answer_marker = "answer is"

    def check_answers_equiv(self, answer_a: str, answer_b: str) -> bool:

        def extract_sql_fragment(raw: str) -> str:
            """
            从 raw 文本中抽取 SQL 片段：
            1) 优先查找 ```sql\n ... \n``` 里边的内容
            2) 找不到的话，就从第一个 SELECT 开始，到第一个 ```（或文本结尾）为止
            3) 去除首尾空白，保留内部格式
            """
            # 1. 尝试提取被 ```sql…``` 包裹的
            m = re.search(r"```sql\s*\n(.+?)\n```", raw, flags=re.S|re.I)
            if m:
                return m.group(1).strip()

            # 2. 否则，找第一个 SELECT 开头
            low = raw.lower()
            idx = low.find("select")
            if idx != -1:
                fragment = raw[idx:]
                # 找到第一个 ``` 分隔符和第一个分号的结束位置
                pos_code = fragment.find("```")
                pos_semi = fragment.find(";")
                if pos_semi != -1:
                    pos_semi += 1  # 包括分号本身
                # 选择最先出现的截断位置
                cut_positions = [p for p in (pos_code, pos_semi) if p != -1]
                if cut_positions:
                    fragment = fragment[:min(cut_positions)]
                return fragment.strip()
            wronginfo='this is not sql.'
            # 3. 万一都没命中，就直接返回原始（去空白）
            return wronginfo

        # 标准化 SQL：提取 ```sql``` 代码块内部或直接使用字符串
        def normalize(sql: str) -> str:

            snippet = extract_sql_fragment(sql).strip().replace('\n', ' ')
            # print('*******')
            # print(snippet)
            # print('*******')
            # dataquery={"query": snippet}
                
            # with open('/root/autodl-tmp/rStar-main2/eval_src/chakan.json', 'a', encoding='utf-8') as f:
            #     json.dump(dataquery, f, ensure_ascii=False, indent=4)
            return snippet
            # 尝试提取 ```sql\n 和 \n``` 之间的内容
            # match = re.search(r"```sql(.*?)```", sql, re.S)
            # if match:
            #     inner = match.group(1)
                
            # else:
            #     inner = sql
                # print('*******')
                # print(inner)
                # print('wrong')
                # print('*******')
            # 保持原始换行和空格，仅去除多余首尾空白
            

        def execute_sql_from_file(db_config, pred_data, gold_data):
            conn = pymysql.connect(
                host=db_config['host'],
                port=db_config['port'],
                user=db_config['user'],
                password=db_config['password'],
                charset='utf8mb4'
            )
            cursor = conn.cursor()
            databaselist=['EnergyManagementDB', 'WaterQualityMonitor', 'generators', 'waterPump', 'Population', 'SmartHomeDB', 'hospital', 'school', 'electric_scooter', 'exerciseclub', 'RainGauge', 'phone', 'architect', 'gas', 'car_engine', 'yacht', 'AlarmSystem', 'contract', 'concert', 'lawnmower', 'printer', 'ECommerce', 'rice_cooker', 'new_energy_vehicles', 'AirCraft', 'PhysicsLabDB', 'car', 'bike', 'mouse']
            flag = False
            for database in databaselist:
                try:
                        cursor.execute(f"USE {database}")  # 切换数据库
                        cursor.execute(pred_data)
                        results1 = cursor.fetchall()

                        cursor.execute(gold_data)
                        results2 = cursor.fetchall()
                        if results2 == results1:
                            dataquery=[]
                            flag=True
                            print('this sql is right!')
                            dataquery.append({"query": pred_data})
                            
                            with open(evaluatefilequeryfile, 'a', encoding='utf-8') as f:
                                json.dump(dataquery, f, ensure_ascii=False, indent=4)
                except Exception as e:
                        # print(f"执行出错: {e}")
                        pass
            

            return flag
        # 示例调用（修改数据库配置和 SQL 文件路径）
        db_config = {
            "host": "sh-cynosdbmysql-grp-noxslim6.sql.tencentcdb.com",
            "port": 26791,
            "user": "root",
            "password": "Ffunkytao110"
        }


        
        filter_anwser_a=normalize(answer_a)
        if answer_a is None or answer_b is None:
            return False
        if filter_anwser_a=='this is not sql.' or filter_anwser_a=='```sql':
            # print('information wrong!')
            return False
        return execute_sql_from_file(db_config, filter_anwser_a, answer_b)

    def check_generator_equiv(self, answer_a: str, answer_b: str) -> bool:

        # 标准化 SQL：去除分号、小写、合并多余空白
        def normalize(sql: str) -> str:
            return re.sub(r"\s+", " ", sql.strip().rstrip(';').lower())

        if answer_a is None or answer_b is None:
            return False
        return normalize(answer_a) == normalize(answer_b)
    def extract_answer_from_gold_solution(self, solution: str) -> str:
        # Gold solution 直接是完整 SQL
        return solution.strip()

    def extract_answer_from_model_completion(self, completion: str) -> str:
        # 利用父类 isolate_answer 方法抽取 SQL
        raw = self.isolate_answer(completion)
        if raw is None:
            return None
        return raw.strip()


class ARCHEREvaluator(Evaluator):
    """
    专用于 Archer 数据集的 Evaluator，进行 SQL 精确匹配评估。
    """
    def __init__(self) -> None:
        super().__init__()
        # 提取答案时，查找 'answer is' 标记
        self.answer_marker = "answer is"

    
    def check_answers_equiv(self, answer_a: str, answer_b: str) -> bool:

        def extract_sql_fragment(raw: str) -> str:
            """
            从 raw 文本中抽取 SQL 片段：
            1) 优先查找 ```sql\n ... \n``` 里边的内容
            2) 找不到的话，就从第一个 SELECT 开始，到第一个 ```（或文本结尾）为止
            3) 去除首尾空白，保留内部格式
            """
            m = re.search(r"```sql\s*\n(.+?)\n```", raw, flags=re.S|re.I)
            if m:
                return m.group(1).strip()

            low = raw.lower()
            idx = low.find("select")
            if idx != -1:
                fragment = raw[idx:]
                pos_code = fragment.find("```")
                pos_semi = fragment.find(";")
                if pos_semi != -1:
                    pos_semi += 1
                cut_positions = [p for p in (pos_code, pos_semi) if p != -1]
                if cut_positions:
                    fragment = fragment[:min(cut_positions)]
                return fragment.strip()
            wronginfo='this is not sql.'
            return wronginfo

        def normalize(sql: str) -> str:
            snippet = extract_sql_fragment(sql).strip().replace('\n', ' ')
            return snippet

        def execute_sql_from_file(filefloader, pred_data, gold_data):
            databaselist = ['bike_1','concert','contract','exerciseclub','formula_1','hospital','school','soccer_1']
            flag = False
            for database in databaselist:
                conn = None
                cursor = None
                db_path = f"{filefloader}/{database}.sqlite"
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()

                    cursor.execute(pred_data)
                    results1 = cursor.fetchall()
                    print('database is ready!')
                    cursor.execute(gold_data)
                    results2 = cursor.fetchall()

                    if results2 == results1:
                        dataquery = [{"query": pred_data}]
                        flag = True
                        print('this sql is right!')

                        with open(evaluatefilequeryfile, 'a', encoding='utf-8') as f:
                            json.dump(dataquery, f, ensure_ascii=False, indent=4)
                        # 如果找到匹配就可以提前返回
                        return True

                except Exception as e:
                    # print(f"执行出错: {e}")
                    pass
                finally:
                    if cursor:
                        cursor.close()
                    if conn:
                        conn.close()
            return flag

        filefloader = '../database/ARCHER'

        filter_anwser_a = normalize(answer_a)
        if answer_a is None or answer_b is None:
            return False
        if filter_anwser_a == 'this is not sql.' or filter_anwser_a == '```sql':
            return False

        return execute_sql_from_file(filefloader, filter_anwser_a, answer_b)

    def extract_answer_from_gold_solution(self, solution: str) -> str:
        # Gold solution 直接是完整 SQL
        return solution.strip()

    def extract_answer_from_model_completion(self, completion: str) -> str:
        # 利用父类 isolate_answer 方法抽取 SQL
        raw = self.isolate_answer(completion)
        if raw is None:
            return None
        return raw.strip()