import os
from sqlalchemy import create_engine
from schema_engine import SchemaEngine
import json
import re
from collections import defaultdict
inputfile='../data/LOGICCAT/all/agent1_output_info_part6.json'
orginfile = '../data/LOGICCAT/all/cop_cot_test_all_augu_merge_part6.json'
outputfile = '../data/LOGICCAT/all/testall_part6.json'

def extractdb(target_db='air_conditioner'):
    # 1. 数据库配置
    db_config = {
        "host": "sh-cynosdbmysql-grp-noxslim6.sql.tencentcdb.com",
        "port": 26791,
        "user": "root",
        "password": "Ffunkytao110"
    }

    # 2. 如果你的 orgin.json 里含有多个 db_id，可以这样筛一次
    with open(orginfile, 'r', encoding='utf-8') as f:
        origin_data = json.load(f)
    # 确保目标库在列表里，否则报错
    if not any(d['db_id'] == target_db for d in origin_data):
        raise ValueError(f"{target_db!r} 不在 orgin.json 中")
    db_name = target_db

    # 3. 构造连接 URL，只连到 air_conditioner
    db_url = (
        f"mysql+pymysql://{db_config['user']}"
        f":{db_config['password']}@"
        f"{db_config['host']}:{db_config['port']}/"
        f"{db_name}"
    )
    engine = create_engine(db_url)

    # 4. 用 SchemaEngine 反射当前连接库
    print(f"开始解析数据库：{db_name} …")
    schema_engine = SchemaEngine(engine=engine, db_name=db_name)
    mschema = schema_engine.mschema

    # 5. 输出或保存
    mschema.save(f'./{db_name}.json')
    print("解析完成。")


def extradb_foreign_keys(tablesfile):
    with open(tablesfile, 'r', encoding='utf-8') as f:
        tablesfile_data = json.load(f)
    db_information = []
    prompt_train_data = []
    # print("**************开始验证外键信息******************")
    for table_data in tablesfile_data:
        db_id = table_data["db_id"]
        table_names = table_data["table_names"]
        column_names = table_data["column_names"]
        foreign_keys = table_data["foreign_keys"]
        primary_keys = table_data["primary_keys"]

        # 构建 table_info
        table_info = {}
        for i, table_name in enumerate(table_names):
            # 获取该表的列名
            columns = [col[1] for col in column_names if col[0] == i]
            table_info[table_name] = ["*"] + columns  # 添加 "*" 作为通用选择

        # 构建 db_foreign_keys
        db_foreign_keys = []

        for fk in foreign_keys:
            for key, value in fk.items():
                # print(f"foreign_key: {key}, foreigntable: {value}")
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

    # print(json.dumps(db_information[0], ensure_ascii=False, indent=2))

    # print(db_information)
    # print()
    # db_information[0]的字典内容为：{'db_name': 'perpetrator', 'table_info': {'perpetrator': ['*', 'Perpetrator_ID', 'People_ID', 'Date', 'Year', 'Location', 'Country', 'Killed', 'Injured'], 'people': ['*', 'People_ID', 'Name', 'Height', 'Weight', 'Home Town']}, 'db_foreign_keys': [{'fk_table': 'perpetrator', 'fk_column': 'People_ID', 'ref_table': 'people', 'ref_column': 'People_ID'}], 'db_primary_keys': [{'pk_table': 'perpetrator', 'pk_column': 'Perpetrator_ID'}, {'pk_table': 'people', 'pk_column': 'People_ID'}]}

    return db_information

#original code
# def extractinfo():
#     with open(inputfile, 'r', encoding='utf-8') as f:
#         input_data = json.load(f)
#     with open(orginfile, 'r', encoding='utf-8') as f:
#         orgin_data = json.load(f)
#     with open('../data/LOGICCAT/db/all_schema2.json', 'r', encoding='utf-8') as f:
#         db_data = json.load(f)

#     parser_grouped_results = []
#     print('开始抽取schema子集')
#     for idx, (datainfo, orgininfo) in enumerate(zip(input_data, orgin_data)):
#         model_output = datainfo['assistant']
#         db_id = orgininfo['db_id']

#         flag = 1
#         # 提取 JSON 块
#         m = re.search(r'```json\s*(\{.*?\})\s*```', model_output, re.DOTALL)
#         if m:
#             json_str = m.group(1)
#         else:
#             json_str = re.sub(r'^\s*```(?:json)?\s*', '', model_output)
#             json_str = re.sub(r'\s*```.*$', '', json_str, flags=re.DOTALL).strip()
#             print(f"第{idx}条模型输出没有找到 JSON 块，跳过处理。")
#             continue  # 跳过无法提取 JSON 的条目
#         try:
#             data = json.loads(json_str)
#         except json.JSONDecodeError as e:
#             print(f"第{idx}条 JSON 解析失败：{e} - 错误数据：{json_str[:100]}")  # 输出错误数据的前100字符
#             # 如果解析失败但是还是会有错误，则将原始数据保存下来
#             parser_grouped_results.append(model_output)
#             continue
#         info = data.get('required_fields', [])

#         data.pop('required_tables', None)
#         data.pop('required_fields', None)
#         filteredorigindata=data
#         # print(filteredorigindata)

#         grouped = defaultdict(list)
#         for f in info:
#             if not isinstance(f, str) or '.' not in f:
#                 continue

#             raw_table, column = f.split('.', 1)
#             full_table = f"{db_id}.{raw_table}"

#             # —— 这里改成 get 链式调用，永远拿到一个 dict 而不会 KeyError ——
#             raw_fields = (
#                 db_data.get('tables', {})
#                 .get(full_table, {})
#                 .get('fields', {})
#                 .get(column, {})
#             )
#             # 过滤不需要的键；raw_fields 可能就是 {}，filtered 也会是 {}
#             filtered = {
#                 k: v for k, v in raw_fields.items()
#                 if k not in {'nullable', 'default', 'autoincrement'}
#             }
#             # column 字段永远都加上
#             col_info = {'column': column, **filtered}
#             grouped[full_table].append(col_info)
#         # 如果 grouped 为空，就输出提示；否则保留分组结果
#         tableinfo = dict(grouped) if grouped else "This sql query does not need table and columns information as it is about mathematical calculations."

#         # ===== 新增：把外键和主键也加进来 =====
#         # 假设你传进来的那个 dict 叫 schema（也可以从 orgininfo 里读）
#         # schema = {
#         #   "db_name": "...",
#         #   "table_info": { … },
#         #   "db_foreign_keys": [ … ],
#         #   "db_primary_keys": [ … ]
#         # }
#         db_information = extradb_foreign_keys('../common/table.json')
#         schema={}
#         for temp in db_information:
#             if temp['db_name'] == db_id:
#                 schema = temp

#         # 1) 拿出当前 query 里用到的原始表名（不含 db 前缀）
#         if isinstance(tableinfo, dict):
#             involved_tables = [t.split('.', 1)[1] for t in tableinfo.keys()]
#         else:
#             involved_tables = []

#         # 2) 过滤出那些涉及到的外键和主键
#         foreign_keys = [
#             fk for fk in schema['db_foreign_keys']
#             if fk['fk_table'] in involved_tables
#         ]
#         primary_keys = [
#             pk for pk in schema['db_primary_keys']
#             if pk['pk_table'] in involved_tables
#         ]

#         # 3) 写入 filteredorigindata
#         filteredorigindata['foreign_keys'] = foreign_keys
#         filteredorigindata['primary_keys'] = primary_keys
#         # ===== 新增结束 =====

#         filteredorigindata.update({'tableinfo':tableinfo})
#         parser_grouped_results.append(filteredorigindata)

#     # print(json.dumps(parser_grouped_results[0], ensure_ascii=False, indent=2))
#     print(f'schema子集抽取成功！数量为：{len(parser_grouped_results)}')
#     # print(len(parser_grouped_results))

#     return parser_grouped_results


def extractinfo():
    with open(inputfile, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    with open(orginfile, 'r', encoding='utf-8') as f:
        orgin_data = json.load(f)
    with open('../data/LOGICCAT/db/all_schema2.json', 'r', encoding='utf-8') as f:
        db_data = json.load(f)

    parser_grouped_results = []
    print('开始抽取schema子集')
    for idx, (datainfo, orgininfo) in enumerate(zip(input_data, orgin_data)):
        model_output = datainfo['assistant']  # 假设这里是模型输出的 JSON 字符串
        db_id = orgininfo['db_id']

        flag = 1
        # if not model_output:
        #         print(f"第{idx}条数据为 None，跳过处理。")
        #         parser_grouped_results.append('there is nothing I can give you about this question.')
        #         continue  # 如果 model_output 为 None，跳过

        # try:
        #         # 尝试直接将 assistant 字符串解析为 JSON
        #         data = json.loads(model_output)
        # except json.JSONDecodeError as e:
        #         print(f"第{idx}条 JSON 解析失败：{e} - 错误数据：{model_output[:100]}")  # 输出错误数据的前100字符
        #         parser_grouped_results.append(model_output)
        #         # 如果有多个 JSON 对象，可以用正则表达式拆分
        #         # match = re.findall(r'\{.*?\}', model_output, re.DOTALL)
        #         # for sub_json_str in match:
        #         #     try:
        #         #         sub_data = json.loads(sub_json_str)
        #         #         # parser_grouped_results.append(sub_data)
        #         #     except json.JSONDecodeError as e:
        #         #         print(f"子 JSON 解析失败：{e} - 错误数据：{sub_json_str[:100]}")
                        

        #         continue
        if not model_output:
             print(f"第{idx}条数据无输出，跳过。")
             parser_grouped_results.append('there is nothing I can give you about this question.')
             continue

         # 处理两种格式：带 ```json ``` Code Fence 或 直接 JSON 字符串
        if '```json' in model_output:
            # 提取 Code Fence 中的 JSON
            m = re.search(r'```json\s*([\s\S]*?)\s*```', model_output)
            if m:
                json_str = m.group(1)
            else:
                print(f"第{idx}条无法定位 JSON Code Fence，跳过。")
                continue
        else:
             # 假设是纯 JSON 字符串
            json_str = model_output.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"第{idx}条 JSON 解析失败：{e} - 数据片段：{json_str[:100]}")
            parser_grouped_results.append('there is nothing I can give you about this question.')
            continue
        

        info = data.get('required_fields', [])
        data.pop('required_tables', None)
        data.pop('required_fields', None)
        filteredorigindata = data

        grouped = defaultdict(list)
        for f in info:
            if not isinstance(f, str) or '.' not in f:
                continue

            raw_table, column = f.split('.', 1)
            full_table = f"{db_id}.{raw_table}"

            # —— 这里改成 get 链式调用，永远拿到一个 dict 而不会 KeyError —— 
            raw_fields = (
                db_data.get('tables', {})
                .get(full_table, {})
                .get('fields', {})
                .get(column, {})
            )
            filtered = {
                k: v for k, v in raw_fields.items()
                if k not in {'nullable', 'default', 'autoincrement'}
            }
            col_info = {'column': column, **filtered}
            grouped[full_table].append(col_info)

        tableinfo = dict(grouped) if grouped else "This sql query does not need table and columns information as it is about mathematical calculations."

        # 处理外键和主键
        db_information = extradb_foreign_keys('../common/table.json')
        schema = {}
        for temp in db_information:
            if temp['db_name'] == db_id:
                schema = temp

        if isinstance(tableinfo, dict):
            involved_tables = [t.split('.', 1)[1] for t in tableinfo.keys()]
        else:
            involved_tables = []

        # 优化外键提取：捕获 KeyError 并继续执行
        try:
            raw_fks = schema['db_foreign_keys']
            if isinstance(raw_fks, list):
                filtered_fks = [fk for fk in raw_fks if fk.get('fk_table') in involved_tables]
                foreign_keys = filtered_fks if filtered_fks else "This issue does not provide foreign keys"
            else:
                foreign_keys = "This issue does not provide foreign keys"
        except KeyError:
            foreign_keys = "This issue does not provide foreign keys"
         # 优化主键提取：捕获 KeyError 并继续执行
        try:
            raw_pks = schema['db_primary_keys']
            if isinstance(raw_pks, list):
                filtered_pks = [pk for pk in raw_pks if pk.get('pk_table') in involved_tables]
                primary_keys = filtered_pks if filtered_pks else "This issue does not provide primary keys"
            else:
                primary_keys = "This issue does not provide primary keys"
        except KeyError:
            primary_keys = "This issue does not provide primary keys"

        filteredorigindata['foreign_keys'] = foreign_keys
        filteredorigindata['primary_keys'] = primary_keys
        filteredorigindata.update({'tableinfo': tableinfo})
        parser_grouped_results.append(filteredorigindata)

    print(f'schema子集抽取成功！数量为：{len(parser_grouped_results)}')
    return parser_grouped_results

def PromptnoMethod(orginfile ,prompt_schema_file, tablesinfo):


    with open(orginfile, 'r', encoding='utf-8') as f:
        trainfile_data = json.load(f)

    # db_information_format=extradb_info(tablesfile)

    prompt_train_data=[]
    questiontype=['normal','physical knowledge','mathematical logic', 'common sense reasoning', 'hypothetical reasoning']
    idx = 0
    for train_data,info in zip(trainfile_data,tablesinfo):
        db_id = train_data['db_id']
        query = train_data['query']
        question = train_data['englishquestion']
        type= int(train_data['type'])
        strinfo = str(info)
        

        # 拼接 Prompt 数据
        prompt_singal_data = {
            'id': idx,
            'problem': (
                    "You are now an expert in SQL statements. "
                    "I will give you information about a database and subsequently ask you a question. All questions involve 4 types of questions: physical knowledge, mathematical logic, common sense reasoning, and hypothetical reasoning. When I ask you a question, I will tell you the type of the question."
                    "Please response me an SQL statement for that question. "
                    "The database information is as follows:\n" + strinfo +"\n"
                    "A "+questiontype[type]+" question is as follows:\n" + question +"\nyour response just have a SQL without anything.\n"

            ),
            'solution': query,
            'db_id': db_id
        }
        idx+=1
        prompt_train_data.append(prompt_singal_data)
        # break
    # print(prompt_train_data[0])

    with open(prompt_schema_file, 'w', encoding='utf-8') as outfile:
        json.dump(prompt_train_data, outfile, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    # extractdb()
    grouped_results=extractinfo()
    # print(f'抽取的schema子集数量为：{len(grouped_results)}')
    PromptnoMethod(orginfile, outputfile, grouped_results)


