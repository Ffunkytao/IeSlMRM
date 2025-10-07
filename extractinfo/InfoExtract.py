import json
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from info_understanding_extractor import extract_info_pipeline

API_KEY = "sk-kecvmweioycupfwhkqinkjrcnnnojhjzptgmrxfgnubzbyed"
BASE_URL = "https://api.siliconflow.cn/v1"
BATCH_SIZE = 1
outputfile = '../data/LOGICCAT/agent1_output_info_500.json'
inputfile = '../data/LOGICCAT/cop_cot_small_english_500.json'
tablefile = '../common/table.json'
# remainfile = '../data/LOGICCAT/agent1_output_info_500_remain.json'

def req(system, user):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1',
        messages=[
            {'role': "system", 'content': system},
            {'role': "user", 'content': user}
        ],
        max_tokens=4096,
        temperature=0.7,
    )
    return resp.choices[0].message.content

def write_record_to_file(record):
    with open(outputfile, 'r+', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
        data.append(record)
        f.seek(0)
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.truncate()

def process_batch(batch, promptdata):
    """并发处理一个 batch，batch 是 [(idx, item), ...]"""
    results = []
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        future_to_idx = {
            executor.submit(req, item['system'], item['user']): idx
            for idx, item in batch
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                assistant = future.result()
            except Exception as e:
                assistant = f"ERROR: {e}"
            system = batch[0][1]['system'] if False else None  # 占位
            user = batch[0][1]['user'] if False else None      # 占位
            # 为了保证顺序，从原 json_data 里取
            record = {
                "idx": idx,
                "system": promptdata[idx]['system'],
                "user": promptdata[idx]['user'],
                "assistant": assistant
            }
            results.append((idx, record))
    return results

def extradb_info(tablesfile):
    with open(tablesfile, 'r', encoding='utf-8') as f:
        tablesfile_data = json.load(f)
    db_information = []
    prompt_train_data = []

    for table_data in tablesfile_data:
        db_id = table_data["db_id"]
        table_names = table_data["table_names"]
        column_names = table_data["column_names"]
        foreign_keys = table_data["foreign_keys"]
        primary_keys = table_data["primary_keys"]

        # print("数据库名称为：")
        # print(db_id)
        # print("***********")
        # print("该数据库的字段为：")
        # print(column_names)
        # print("***********")
        # print("该数据库的表名为：")
        # print(table_names)
        # print("***********")
        # print("该数据库的外键为：")
        # print(foreign_keys)
        #
        # print("***********")
        # print("该数据库的主键为：")
        # print(primary_keys)

        # 构建 table_info
        table_info = {}
        for i, table_name in enumerate(table_names):
            # 获取该表的列名
            columns = [col[1] for col in column_names if col[0] == i]
            table_info[table_name] = ["*"] + columns  # 添加 "*" 作为通用选择

        # 构建 db_foreign_keys
        db_foreign_keys = []
        print("**************开始验证外键信息******************")
        for fk in foreign_keys:
            for key, value in fk.items():
                print(f"foreign_key: {key}, foreigntable: {value}")
                foreign_table = value[0]
                reference_table = value[1]
                # print(f"Table {table_names[foreign_table]} FOREIGN KEY ({key}) REFERENCES Table {table_names[reference_table]} ({key}) ")
                db_foreign_keys.append({
                    'fk_table': table_names[foreign_table],
                    'fk_column': key,
                    'ref_table': table_names[reference_table],
                    'ref_column': key
                })
                # 遍历外键列表
        # for fk in table_data['foreign_keys']:
        #     for fk_column, (fk_col_index, ref_col_index) in fk.items():
        #             # 获取外键字段的表索引和列名
        #             fk_column_index = fk_col_index
        #             fk_column_name = column_names[fk_column_index][1]
        #             fk_table_index = column_names[fk_column_index][0]
        #             fk_table_name = table_names[fk_table_index]
        #
        #             # 获取引用字段的表索引和列名
        #             ref_column_index = ref_col_index
        #             ref_column_name = column_names[ref_col_index][1]
        #             ref_table_index = column_names[ref_column_index][0]
        #             ref_table_name = table_names[ref_table_index]
        #             # fk_column_index, ref_column_index = fk
        #             #
        #             # # 获取外键字段的表名和列名
        #             # fk_table_index = colunmns[fk_column_index][0]
        #             # fk_column_name = colunmns[fk_column_index][1]
        #             # fk_table_name = table_names[fk_table_index]
        #             #
        #             # # 获取引用字段的表名和列名
        #             # ref_table_index = colunmns[ref_column_index][0]
        #             # ref_column_name = colunmns[ref_column_index][1]
        #             # ref_table_name = table_names[ref_table_index]
        #             foreign_keys.append({
        #                 'fk_table': fk_table_name,
        #                 'fk_column': fk_column_name,
        #                 'ref_table': ref_table_name,
        #                 'ref_column': ref_column_name
        #             })

        # 构建 db_primary_keys
        db_primary_keys = []
        for pk in primary_keys:
            for pk_column, pk_table_idx in pk.items():
                pk_table = table_names[pk_table_idx]
                db_primary_keys.append({
                    "pk_table": pk_table,
                    "pk_column": pk_column
                })

        # 构建最终结果
        result = {
            "db_name": db_id,
            "table_info": table_info,
            "db_foreign_keys": db_foreign_keys,
            "db_primary_keys": db_primary_keys
        }
        db_information.append(result)

    # for item in db_information:
    #     if item['db_name'] == 'e_commerce':
    #         print(item)

    # with open('fixdata/tableview.json', 'w', encoding='utf-8') as outfile:
    #     json.dump(db_information, outfile, ensure_ascii=False, indent=4)

    # print(db_information)
    # print(db_information[0])
    # db_information[0]的字典内容为：{'db_name': 'perpetrator', 'table_info': {'perpetrator': ['*', 'Perpetrator_ID', 'People_ID', 'Date', 'Year', 'Location', 'Country', 'Killed', 'Injured'], 'people': ['*', 'People_ID', 'Name', 'Height', 'Weight', 'Home Town']}, 'db_foreign_keys': [{'fk_table': 'perpetrator', 'fk_column': 'People_ID', 'ref_table': 'people', 'ref_column': 'People_ID'}], 'db_primary_keys': [{'pk_table': 'perpetrator', 'pk_column': 'Perpetrator_ID'}, {'pk_table': 'people', 'pk_column': 'People_ID'}]}

    db_information_format = []
    # 格式化的db_information_format，存储了一堆格式化的数据表信息。
    # "First: perpetrator database has perpetrator, people  tables.\n
    # Second: below are columns of table, primary_keys of table:\n
    # 1 The perpetrator table has  Perpetrator_ID, People_ID, Date, Year, Location, Country, Killed, Injured columns, and Perpetrator_ID is primary_key.\n
    # 2 The people table has People_ID, Name, Height, Weight, Home Town, and People_ID is primary_key.\n
    # Third: below are foreign_keys  of tables :\n
    # perpetrator.People_ID to people.People_ID.\n"

    for dict1 in db_information:
        # 初始化一个空列表，用于存储每一行的字符串
        lines = []
        # db_information[0]就是dict
        # 第一部分: 数据库名称和表名
        db_name = dict1.get('db_name', 'Unknown')
        tables = ', '.join(dict1['table_info'].keys())
        first_part = f"First: {db_name} database has {tables} tables.\n"
        lines.append(first_part)

        # 第二部分: 表的列和主键信息
        second_part = "Second: below are columns of table, primary_keys of table:\n"
        lines.append(second_part)

        table_info = dict1.get('table_info', {})
        primary_keys = {pk['pk_table']: pk['pk_column'] for pk in dict1.get('db_primary_keys', [])}
        # if dict1['db_name']=='e_commerce':
        #     print(primary_keys)

        for idx, (table, columns) in enumerate(table_info.items(), start=1):
            # 排除 '*' 并去除重复的列（如果有）
            filtered_columns = [col for col in columns if col != '*']
            columns_str = ', '.join(filtered_columns)

            # 获取当前表的主键
            pk = primary_keys.get(table, 'Unknown')
            # if dict1['db_name'] == 'e_commerce':
            #     print("****")
            #     print(table)
            #     print(primary_keys)
            #     print(pk)
            # # if pk=='Unknown':
            #     print(table_info)
            if pk != 'Unknown':
                table_line = f"{idx} The {table} table has {columns_str} columns, and {pk} is primary_key.\n"
            else:
                table_line = f"{idx} The {table} table has {columns_str} columns, and did not have primary_key.\n"
            lines.append(table_line)

        # 第三部分: 外键信息
        third_part = "Third: below are foreign_keys of tables:\n"
        lines.append(third_part)

        foreign_keys = dict1.get('db_foreign_keys', [])
        for fk in foreign_keys:
            fk_table = fk.get('fk_table', 'Unknown')
            fk_column = fk.get('fk_column', 'Unknown')
            ref_table = fk.get('ref_table', 'Unknown')
            ref_column = fk.get('ref_column', 'Unknown')

            fk_line = f"{fk_table}.{fk_column} to {ref_table}.{ref_column}.\n"
            lines.append(fk_line)

        # 将所有行合并成一个字符串
        str1 = ''.join(lines)
        # 输出结果
        # print(str1)

        dicttableinfo = {
            'db_id': db_name,
            'db_info': str1
        }
        db_information_format.append(dicttableinfo)

    # for item in db_information_format:
    #     if item['db_id'] == 'e_commerce':
    #         print(item)

    print(db_information_format[0])
    # db_information_format的每个字典内容为：
    # {'db_id': 'perpetrator', 'db_info': 'First: perpetrator database has perpetrator, people tables.\nSecond: below are columns of table, primary_keys of table:\n1 The perpetrator table has Perpetrator_ID, People_ID, Date, Year, Location, Country, Killed, Injured columns, and Perpetrator_ID is primary_key.\n2 The people table has People_ID, Name, Height, Weight, Home Town columns, and People_ID is primary_key.\nThird: below are foreign_keys of tables:\nperpetrator.People_ID to people.People_ID.\n'}

    return db_information_format

def datatoprompt(json_data,table_data):

    origin_data = json_data

    db_information_format=extradb_info(table_data)

    prompt_train_data=[]
    idx = 0

    for train_data in origin_data:
        db_id = train_data['db_id']

        question = train_data['englishquestion']

        strinfo = ''
        example= "\n{\n  \"entities\": [\"air conditioner model AC-1234\", \"indoor temperature 30°C\", \"target temperature 25°C\"],\n  \"relations\": [\"model_to_specs (AC-1234 → technical specifications)\", \"temperature_difference (ΔT=5°C)\"],\n  \"intent\": \"calculate_energy_required_for_cooling\",\n  \"reasoning_type\": \"physical knowledge reasoning\",\n  \"numerical_values\": [30, 25],\n  \"units\": {\n    \"temperature\": \"°C (Celsius)\",\n    \"energy\": \"kWh (implied by database schema)\",\n    \"conversion\": \"ΔT = 5°C (temperature differential)\"\n  },\n  \"required_tables\": [\"air_conditioner_info\", \"energy_consumption\"],\n  \"required_fields\": [\n    \"air_conditioner_info.model\", \n    \"air_conditioner_info.cooling_capacity_btu\",\n    \"air_conditioner_info.energy_efficiency_ratio\",\n    \"energy_consumption.power_consumption_watts\",\n    \"energy_consumption.energy_consumption_kwh\"\n  ]\n}"
        for data in db_information_format:
            if db_id == data['db_id']:
                strinfo = data['db_info']

        # 拼接 Prompt 数据
        prompt_singal_data = {
            'system': (
                    "You are a Text-to-SQL intelligence and one of your sub-tasks is to understand a natural language processing question posed by a user and help me extract the exact entities, "
                    "relationships, intents, reasoning types, values, units, tables and fields to be used for the question. Entity refers to the key information asked by the user, such as the query object; "
                    "Relationship refers to the relationship between entities; Intent is the intention of the user's query; "
                    "Reasoning type is divided into: physical knowledge reasoning, mathematical logic reasoning, common sense understanding reasoning, "
                    "ideal hypothesis reasoning. Numerical value refers to the specific parameter value requested by that query; Unit refers to the unit conversion formula, unit type, unit value, etc. "
                    "for the calculation of that question. I will give you a user's natural language question, please answer me in Json format and The key value of your json should be qualified as entities, "
                    "relations, intent, reasoning_type, numerical_values, units, required_tables, and required_fields. "
                    "this is example:"+example+"\n"
                    "The question of user is as follows:\n" + question +"\n"

            ),
            'user': "The database information is as follows:"+strinfo,
            'assistant': '',
            'db_id': db_id
        }
        idx+=1
        prompt_train_data.append(prompt_singal_data)

    return prompt_train_data


def main():

    print('loading datasets...')

    # 先载入所有数据
    with open(inputfile, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    promptdata = datatoprompt(json_data, tablefile)

    print('infering table info and intent...')

    # print(promptdata)

    # 按 batch 并发
    for start in tqdm(range(0, len(promptdata), BATCH_SIZE), desc="Batches"):
        batch = list(enumerate(promptdata[start:start + BATCH_SIZE], start=start))

        batch_results = process_batch(batch, promptdata)
        # 按 idx 升序写回文件
        for _, record in sorted(batch_results, key=lambda x: x[0]):
            write_record_to_file(record)

    print("infering finished...")

def extract_data_remain():
    # 先载入所有数据
    with open(outputfile, 'r', encoding='utf-8') as f:
        output_data = json.load(f)
    templist = []

    for data in output_data:
        if data['assistant'] == '':
            templist.append(data)
    with open(remainfile, 'w', encoding='utf-8') as outfile:
        json.dump(templist, outfile, ensure_ascii=False, indent=4)

def merge():
    with open(outputfile, 'r', encoding='utf-8') as f:
        output_data = json.load(f)

    with open('../data/LOGICCAT/agent1_output_info_500_fix.json', 'r', encoding='utf-8') as f:
        output_data1 = json.load(f)

    with open('../data/LOGICCAT/agent1_output_info_500_remain.json', 'r', encoding='utf-8') as f:
        output_data2 = json.load(f)
    templist = output_data

    for data1,data2 in zip(output_data1, output_data2):
        idx = data2['idx']
        templist[idx]['assistant'] = data1['assistant']

    print(templist)

    with open('../data/LOGICCAT/agent1_output_result.json', 'w', encoding='utf-8') as outfile:
        json.dump(templist, outfile, ensure_ascii=False, indent=4)

def verification():

    # with open('../data/LOGICCAT/agent1_output_info_500_fix.json', 'r', encoding='utf-8') as f:
    #     output_data1 = json.load(f)

    # with open('../data/LOGICCAT/agent1_output_info_500_remain.json', 'r', encoding='utf-8') as f:
    #     output_data2 = json.load(f)

    with open('../data/LOGICCAT/agent1_output_info_500_fix.json', 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    idx=0
    prompt_train_data=[]
    for data in data1:
        idx+=1
        question = data1['question']
        strinfo = data1['db_info']

        promptdata = datatoprompt()

        extract_result = extract_info_pipeline(
            question=question,           # train_data['englishquestion']
            db_info_text=strinfo,        # extradb_info() 格式化后的 schema 信息
            llm_fn=req,                  # 直接复用你的 LLM 请求函数
            delta_match=0.6,             # δ_match
            tau=0.7                      # τ
        )

        # 把 extract_result 合并进最终写出的 record
        record = {
            "idx": idx,
            "system": promptdata[idx]['system'],
            "user": promptdata[idx]['user'],
            "assistant": '',          # 原始LLM草稿JSON文本
            "info_extraction": extract_result  # 完整管线产物：(R_tuple, R_init, R_cand, plans, scores, R_high)
        }

        prompt_train_data.append(record)



    return prompt_train_data

def extractdata():
    templist=verification()
    with open('../data/LOGICCAT/agent1_output_result.json', 'w', encoding='utf-8') as outfile:
        json.dump(templist, outfile, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    main()
    # extract_data()
    # merge()
    # datatoprompt
    # datatest=verification()
    # print(datatest[0])
    # extractdata()

