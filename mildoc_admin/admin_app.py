# -*- coding: utf-8 -*-

# =============================================第一层：基础和配置============================================
# 导入模块、加载环境变量与配置常量、Flask 实例化与密钥配置、

# 从Flask导入Web开发所需的各种核心函数和类
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, abort
from minio import Minio             # 导入MinIO客户端，MinIO是一个用来存文件的仓库（类似私有化版百度网盘）
from pymilvus import MilvusClient   # 导入Milvus客户端，Milvus是一个专门存大模型向量数据的数据库
import os
from dotenv import load_dotenv
from datetime import timezone       # 导入时区处理模块
from functools import wraps         # 导入装饰器工具，用来写登录验证等功能
import pytz                         # 导入更强大的时区处理库，用来把外国时间转成北京时间


# 加载环境变量
load_dotenv()



# 管理员账号
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

# Minio 配置
MINIO_BUCKET = os.getenv('MINIO_BUCKET')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT')        # 读取MinIO服务器的IP地址和端口
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
MINIO_REGION = os.getenv('MINIO_REGION')            # 读取MinIO所在的数据中心地区
MINIO_USE_VIRTUAL_HOST = os.getenv('MINIO_USE_VIRTUAL_HOST', 'false').lower() == 'true'   # 读取是否用域名方式访问，默认关掉
MINIO_USE_SSL = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'      # 读取是否开启HTTPS加密传输，默认关掉

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST")  # 读取Milvus数据库的IP地址
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_USER = os.getenv("MILVUS_USER")
MILVUS_PASSWORD = os.getenv("MILVUS_PASSWORD")
MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")       # 读取要连哪个数据库（类似于MySQL里的database）
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")   # 读取要操作哪张“表”（Milvus里叫集合Collection，存文件分块向量的地方）
MILVUS_INDEX_NAME = os.getenv("MILVUS_INDEX_NAME")   # 读取向量的索引名称（索引就像字典的目录，能加快搜索速度）




app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key') # 设置加密盐，用来给用户的登录状态（Session）加密防篡改





# =============================================第二层：核心组件============================================
# 初始化第三方客户端（Minio, Milvus）、请求钩子 (before_request)、内部辅助函数 (_get_minio_client)、
# 业务工具函数 (convert_utc_to_local)、自定义装饰器 (login_required)、初始化第三方客户端（Minio, Milvus）

@app.before_request  # Flask 请求钩子
def before_request():
    """在每个请求之前执行域名白名单检查"""
    app.logger.info(f"before_request: {request.url}")

# 内部函数，连接MinIO服务器的客户端对象
def _get_minio_client() -> Minio:
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_USE_SSL,   # 是否开启HTTPS加密传输，默认关掉
        region=MINIO_REGION,    # 读取MinIO所在的数据中心地区
    )

    if MINIO_USE_VIRTUAL_HOST:
        client.enable_virtual_style_endpoint()
        
    return client

# 初始化 Minio 客户端，全局可用
minio_client = _get_minio_client()


# 初始化 Milvus API
milvus_client = MilvusClient(
    uri=f"http://{MILVUS_HOST}:{MILVUS_PORT}",  # 读取Milvus数据库的IP地址和端口，拼接网址
    token=f"{MILVUS_USER}:{MILVUS_PASSWORD}",   # 拼接账号密码作为令牌
    db_name=MILVUS_DATABASE                     # 指定连哪个库
)


# 时间转换函数，把服务器标准时间转成北京时间
def convert_utc_to_local(utc_datetime, timezone_name='Asia/Shanghai'):
    """
    将UTC时间转换为本地时间
    
    Args:
        utc_datetime: UTC时间对象（datetime）
        timezone_name: 目标时区名称，默认为'Asia/Shanghai'（中国标准时间）
    
    Returns:
        str: 格式化后的本地时间字符串（YYYY-MM-DD HH:MM:SS）
    """
    if not utc_datetime:
        return ''
    
    try:
        # 确保UTC时间有时区信息
        if utc_datetime.tzinfo is None:  # 如果这个时间没有带时区标签
            utc_time = utc_datetime.replace(tzinfo=timezone.utc) # 强行给它贴上“UTC时区”的标签
        else: # 如果它本来就带时区标签
            utc_time = utc_datetime # 直接用原时间
        
        # 转换为目标时区
        local_tz = pytz.timezone(timezone_name) # 找到“亚洲/上海”这个时区的规则
        local_time = utc_time.astimezone(local_tz) # 把UTC时间转换为“亚洲/上海”时区的时间
        
        # 格式化为字符串，时间变成 "2023-10-01 12:00:00" 
        return local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    except Exception as e:
        app.logger.error(f"时区转换失败: {str(e)}")
        # 如果转换失败，返回原始时间的字符串格式
        return utc_datetime.strftime('%Y-%m-%d %H:%M:%S') if utc_datetime else ''

# 定义一个装饰器，用来拦截没登录的用户
def login_required(f):
    """登录验证装饰器"""
    @wraps(f) # 这个魔法能让被装饰的函数不丢失自己的名字等信息
    def decorated_function(*args, **kwargs): # 定义真正用来拦截的内部函数
        if 'username' not in session: # 检查用户的Session里有没有用户名，没登录就没有
            return redirect(url_for('login')) # 如果没有，强制把他踢到登录页面
        return f(*args, **kwargs) # 如果有，放行，执行原本要执行的页面函数
    return decorated_function # 返回这个包装好的拦截器


# =============================================第三层：页面视图路由============================================
# 9. index()首页跳转、10. login()登录页、11. logout()退出、12. file_browser()文件列表页、13. file_detail()文件详情页


@app.route('/') # 绑定根目录
def index(): # 首页处理函数
    """首页，重定向到登录页面或文件浏览页面"""
    if 'username' in session:   # 用户已登录
        return redirect(url_for('file_browser')) # 重定向（跳转）到文件浏览页面
    return redirect(url_for('login')) # 未登录则重定向到登录页面

@app.route('/login', methods=['GET', 'POST']) # 绑定登录页网址，允许GET（看页面）和POST（提交表单）两种请求
def login(): # 登录页处理函数
    """登录页面"""
    if request.method == 'POST': # 如果用户是点击了“登录”按钮提交过来的
        username = request.form['username']
        password = request.form['password']
        
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['username'] = username
            flash('登录成功！', 'success')
            return redirect(url_for('file_browser')) # 跳转到文件浏览页
        else:
            flash('用户名或密码错误', 'error') # 给页面发一个红色的错误提示条
    
    return render_template('login.html') # 不管是刚进登录页，还是密码错了，都渲染出登录的HTML页面

@app.route('/logout') # 绑定退出登录的网址
def logout():
    """退出登录"""
    session.pop('username', None) # 把Session里的用户名删掉，代表他下线了
    flash('已退出登录', 'info')
    return redirect(url_for('login')) # 踢回登录页


# 企业微信登录相关接口，暂时不使用
#app.route('/wxwork/callback')

#app.route('/wxwork/login')

#app.route('/wxwork/verify', methods=['GET', 'POST'])

#app.route('/api/wxwork/user')

# 文件解析测试 API，暂时不使用
#app.route('/api/file/<path:file_path>/parse-test', methods=['POST'])


# 文件浏览页面
@app.route('/files')
@login_required # 登录拦截
def file_browser():
    """文件浏览页面"""
    path = request.args.get('path', '') # 从网址参数里获取当前要看哪个文件夹路径，没有就默认空（根目录）
    return render_template('file_browser.html', current_path=path) # 渲染出文件浏览页的HTML页面，把当前路径也传给它


# 获取文件列表 API
@app.route('/api/files') # 绑定获取文件列表的接口（前端用Ajax偷偷调用的）
@login_required
def api_files():
    """获取文件列表 API"""
    path = request.args.get('path', '') # 获取前端传来的当前路径
    
    try:
        # 测试 MinIO 连接
        if not minio_client.bucket_exists(MINIO_BUCKET): # 先看看配置的bucket在MinIO里存不存在
            return jsonify({'error': f'桶 "{MINIO_BUCKET}" 不存在，请检查配置'}), 500
        
        # 获取 Minio 中的文件列表
        # 构建要搜索的路径前缀，确保路径以“/”结尾
        search_prefix = path if path == '' else (path if path.endswith('/') else path + '/')
        
        # 去MinIO查这个路径下的内容，recursive=False表示不查子文件夹里的东西
        objects = minio_client.list_objects(MINIO_BUCKET, prefix=search_prefix, recursive=False)
        
        files = [] # 装一会要返回给前端的文件信息
        folders = set() # 自动去重，装一会找到的文件夹名
        
        # 遍历查出来的每一个对象（可能是文件，也可能是文件夹的标记）
        for obj in objects:
            # 跳过当前目录的标记对象
            if obj.object_name == search_prefix.rstrip('/'): # 如果这个对象名字正好就是当前路径本身（MinIO用空文件代表文件夹）
                continue    # 跳过它不展示
            
            # 计算相对路径
            if search_prefix:
                if not obj.object_name.startswith(search_prefix): # 如果对象名不以该前缀开头
                    continue
                relative_path = obj.object_name[len(search_prefix):] # 把前缀截掉，剩下就是相对路径，比如 "a/b.txt" -> "b.txt"
            else: # 如果没有前缀（说明在根目录）
                relative_path = obj.object_name # 相对路径就是它自己的全名
            
            # 跳过空路径
            if not relative_path: # 如果截完是空的就跳过
                continue
            
            if '/' in relative_path: # 如果相对路径里还有斜杠，说明它是某个子文件夹里的文件
                # 这是一个子目录中的对象，我们只需要目录名
                folder_name = relative_path.split('/')[0] # 把斜杠劈开，取第一段作为文件夹名
                if folder_name and folder_name not in folders: # 如果文件夹名不是空的，且不在集合里
                    folders.add(folder_name) # 记到集合里
                    folder_path = path + '/' + folder_name if path else folder_name # 拼出这个文件夹的完整路径
                    files.append({
                        'name': folder_name,
                        'path': folder_path,
                        'type': 'folder',       # 标记类型为文件夹
                        'size': 0,              # 文件夹大小算0
                        'modified': ''          # 修改时间留空
                    })
            # 路径无斜杠则为文件
            else:
                # 这是当前目录下的文件
                files.append({
                    'name': relative_path,
                    'path': obj.object_name,    # 文件完整路径
                    'type': 'file',             # 标记类型为文件
                    'size': obj.size,           # 文件大小
                    # 转换为本地时间格式
                    'modified': convert_utc_to_local(obj.last_modified)
                })
        
        return jsonify({'files': files})    # 把整理好的列表变成JSON格式返回给前端
    except Exception as e:
        app.logger.error(f"获取文件列表失败: {str(e)}")
        if "SignatureDoesNotMatch" in str(e):   # 认证失败
            return jsonify({'error': 'MinIO 认证失败，请检查 ACCESS_KEY 和 SECRET_KEY 配置'}), 500
        elif "Connection" in str(e):    # 连接失败
            return jsonify({'error': f'无法连接到 MinIO 服务器 ({os.getenv("ENDPOINT")})，请检查网络和地址配置'}), 500
        else:
            return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500

# 文件详情页面
@app.route('/file/<path:file_path>')
@login_required
def file_detail(file_path):
    """文件详情页面"""
    return render_template('file_detail.html', file_path=file_path) # 渲染HTML页面，把文件路径传过去



# =============================================第四层：API 接口路由（按“资源”分组聚拢）============================================
# 14. 【文件资源 API 组】
# api_files()获取文件列表（这个因为开发流程所以在上一层）、api_file_detail()获取文件详情、api_upload_files()上传文件、api_download_file()下载文件、api_delete_file()删除文件
# 15. 【目录资源 API 组】
# api_create_directory()创建目录、api_delete_directory()删除目录


# 获取文件详情 API
@app.route('/api/file/<path:file_path>') # 绑定获取文件详情的接口
@login_required
def api_file_detail(file_path):
    """获取文件详情 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote # 导入URL解码工具
        # 先尝试解码一次，如果结果仍然是编码格式，再解码一次
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path
        
        app.logger.debug(f"原始路径: {file_path}")  # 绝对路径
        app.logger.debug(f"解码后路径: {decoded_path}")
        # 从 Minio 获取文件基本信息
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path) # 获取文件的元数据（大小、修改时间等，不下载文件本身）
            
            file_info = {
                'doc_name': os.path.basename(file_path), # 取出纯文件名（去掉了前面的文件夹路径）
                'doc_path_name': file_path, # 文件完整路径
                'file_size': file_stat.size,
                'file_md5': file_stat.etag.strip('"'), # 文件的MD5校验码
                'last_modified': convert_utc_to_local(file_stat.last_modified), # 文件的最后修改时间
                'doc_type': '', # 文件类型
                'indexed': False,   # 默认未被嵌入，即 Milvus里没查到
                'index_time': '',   # 嵌入时间留空
                'chunks': []        # 分片信息为空
            }
        except Exception as e:
            app.logger.error(f"获取文件信息失败: {str(e)}")
            if "SignatureDoesNotMatch" in str(e): # 密码错
                return jsonify({'error': 'MinIO 认证失败，请检查 ACCESS_KEY 和 SECRET_KEY 配置'}), 500
            elif "NoSuchKey" in str(e) or "not found" in str(e).lower(): # 文件不存在
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 从 Milvus 查询文件索引信息
        try:
            milvus_client.load_collection(collection_name=MILVUS_COLLECTION) # 把存向量的“表”加载到内存里（因为数据大，不加载查不了）
            filter_expr = f'doc_path_name == "{file_path}"'  # 写一句过滤条件：找路径等于当前路径的数据
            # 示例：
            # filter_expr = 'doc_path_name == "/documents/report.pdf"'
            # 结果只返回 id 为 1、2、3 的三行，因为只有这三行满足路径条件


            results = milvus_client.query(
                collection_name=MILVUS_COLLECTION,
                filter=filter_expr,     # 指定过滤条件
                output_fields=["doc_name", "doc_path_name", "doc_type", "doc_md5", "doc_length", "content", "embedding_model"], # 指定要查的列
                limit=100               # 最多查100条（切片）
            )
            
            if results:     # 如果查到了数据
                file_info['indexed'] = True # 把状态改成“已向量化”
                # 取第一条记录的基本信息
                first_record = results[0]
                file_info['doc_type'] = first_record.get('doc_type', '') # 文件类型
                file_info['indexed_md5'] = first_record.get('doc_md5', '') # 文件的MD5校验码
                file_info['doc_length'] = first_record.get('doc_length', 0) # 文件的总长度（字符数），Milvus 集合中，同一个文件的所有切片记录都冗余存储了原始文件信息
                file_info['embedding_model'] = first_record.get('embedding_model', '') # 用的哪个嵌入模型
                
                # 处理所有分片
                for result in results:
                    file_info['chunks'].append({ # 把每一段的文本塞进列表
                        'content': result.get('content', ''), # 分片内容
                        'length': len(result.get('content', '')) # 分片长度
                    })
        except Exception as e:
            print(f"查询 Milvus 失败: {e}")
        
        return jsonify(file_info) # 把最终组装好的文件详情返回给前端
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 文件下载 API
@app.route('/api/file/<path:file_path>/download')   # 绑定下载接口
@login_required
def api_download_file(file_path):
    """文件下载 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path
        
        app.logger.info(f"下载文件: {file_path}")
        
        # 检查文件是否存在
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path) # 查文件信息
        except Exception as e:
            if "NoSuchKey" in str(e) or "not found" in str(e).lower(): # 文件不存在
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 从 MinIO 获取文件
        from flask import Response # 导入Flask的响应对象，用来构造下载流
        import io # 导入io流处理库
        
        try:
            response = minio_client.get_object(MINIO_BUCKET, file_path) # 从MinIO把文件数据抽出来，返回一个响应对象
            file_data = response.data # 数据读入内存
            
            # 获取文件名
            filename = os.path.basename(file_path)
            
            # 设置正确的 Content-Type
            content_type = file_stat.content_type or 'application/octet-stream' # 获取文件MIME类型，不知道就默认当二进制流
            
            # 对文件名进行URL编码以支持中文字符
            from urllib.parse import quote # 导入URL编码函数
            encoded_filename = quote(filename, safe='') # 把中文文件名编码，防止浏览器下载时乱码
            
            # 创建响应
            return Response(
                file_data,  # 塞入文件二进制数据
                mimetype=content_type, # 设置文件类型
                headers={ # 设置响应头
                    'Content-Disposition': f'attachment; filename*=UTF-8\'\'{encoded_filename}', # 表面附件需要下载，文件名是编码后的
                    'Content-Length': str(len(file_data)), # 文件大小，可用来显示进度条
                    'Cache-Control': 'no-cache' # 不需要缓存，每次重新下载即可
                }
            )
            
        except Exception as e:
            app.logger.error(f"下载文件失败: {str(e)}")
            return jsonify({'error': f'下载文件失败: {str(e)}'}), 500
        finally:
            if 'response' in locals(): # 如果前面创建了response对象
                response.close()   # 关闭网络连接
                response.release_conn() # 把连接还给连接池
        
    except Exception as e:
        app.logger.error(f"下载请求处理失败: {str(e)}")
        return jsonify({'error': f'下载请求处理失败: {str(e)}'}), 500

# 文件删除 API
@app.route('/api/file/<path:file_path>/delete', methods=['DELETE'])  # 绑定删除接口，只接受DELETE请求
@login_required
def api_delete_file(file_path):
    """文件删除 API"""
    try:
        # URL 解码文件路径（处理可能的双重编码）
        from urllib.parse import unquote
        decoded_path = unquote(file_path)
        if '%' in decoded_path:
            decoded_path = unquote(decoded_path)
        file_path = decoded_path
        
        app.logger.info(f"删除文件: {file_path}")
        
        # 检查文件是否存在
        try:
            file_stat = minio_client.stat_object(MINIO_BUCKET, file_path)
            file_name = os.path.basename(file_path)  # 提取纯文件名，等会提示用
        except Exception as e:
            if "NoSuchKey" in str(e) or "not found" in str(e).lower():
                return jsonify({'error': f'文件不存在: {file_path}'}), 404
            else:
                return jsonify({'error': f'获取文件信息失败: {str(e)}'}), 500
        
        # 从 MinIO 删除文件，即处理请求
        try:
            minio_client.remove_object(MINIO_BUCKET, file_path) # 调用MinIO接口把文件删了
            app.logger.info(f"文件删除成功: {file_path}")
            
            # 如果文件在 Milvus 中有索引，也删除索引记录
            try:
                milvus_client.load_collection(collection_name=MILVUS_COLLECTION) # 加载向量表
                filter_expr = f'doc_path_name == "{file_path}"' # 拼过滤条件：找这个路径对应的向量
                
                # 查询是否存在索引记录
                results = milvus_client.query(
                    collection_name=MILVUS_COLLECTION,  # 表名
                    filter=filter_expr,     # 过滤条件
                    output_fields=["id"],   # 只查主键ID
                    limit=1000              # 最多查1000条
                )
                
                if results:
                    # 删除 Milvus 中的记录
                    ids_to_delete = [str(result['id']) for result in results] # 把所有的id提取出来变成字符串列表
                    milvus_client.delete( # 执行删除
                        collection_name=MILVUS_COLLECTION,
                        filter=f'id in {ids_to_delete}' # 拼出 "id in [1, 2, 3]" 这样的条件来删
                    )
                    app.logger.info(f"已删除 Milvus 中的 {len(ids_to_delete)} 条记录")
                
            except Exception as e:
                app.logger.warning(f"删除 Milvus 索引时出错 (文件已删除): {str(e)}")
            
            return jsonify({
                'success': True,
                'message': f'文件 "{file_name}" 删除成功',
                'file_name': file_name, 
                'file_path': file_path
            })
            
        except Exception as e:
            app.logger.error(f"删除文件失败: {str(e)}")
            return jsonify({'error': f'删除文件失败: {str(e)}'}), 500
        
    except Exception as e:
        app.logger.error(f"删除请求处理失败: {str(e)}")
        return jsonify({'error': f'删除请求处理失败: {str(e)}'}), 500

# 创建目录 API
@app.route('/api/create-directory', methods=['POST']) # 绑定创建文件夹接口
@login_required
def api_create_directory():
    """创建目录 API"""
    try:
        data = request.get_json() # 从前端获取传过来的JSON数据
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        current_path = data.get('path', '')     # 获取要在哪个路径下创建
        dir_name = data.get('name', '').strip() # 获取新文件夹的名字，并去掉两头空格
        
        # 验证目录名称
        if not dir_name:
            return jsonify({'success': False, 'error': '目录名称不能为空'}), 400
        
        # 检查非法字符
        import re  # 导入正则表达式库
        if re.search(r'[/\\:*?"<>|]', dir_name): # 如果名字里包含这些特殊符号
            return jsonify({'success': False, 'error': '目录名称不能包含以下字符: / \\ : * ? " < > |'}), 400
        
        # 构建完整路径
        if current_path: # 如果有父路径
            full_path = current_path.rstrip('/') + '/' + dir_name + '/' # 拼成 "父路径/新文件夹名/" 的格式
        else:
            full_path = dir_name + '/' # 根目录建
        
        app.logger.info(f"创建目录: {full_path}")
        
        # 检查目录是否已存在
        try:
            minio_client.stat_object(MINIO_BUCKET, full_path) # 去查这个路径存不存在
            return jsonify({'success': False, 'error': '目录已存在'}), 400
        except Exception:
            # 目录不存在，可以创建
            pass
        
        # 创建目录（通过上传一个空对象）
        from io import BytesIO # 导入内存流工具，用于创建空对象
        minio_client.put_object( # 调用上传接口
            MINIO_BUCKET,
            full_path,
            BytesIO(b''), # 上传一个空对象，内容为空。因为是目录嘛
            0,  # 上传大小为0，因为是目录嘛
            content_type='application/x-directory' # 假装它是一个目录类型的文件（MinIO本身没有真文件夹，都是靠这种空对象模拟的）
        )
        
        app.logger.info(f"目录创建成功: {full_path}")
        return jsonify({'success': True, 'message': '目录创建成功'})
        
    except Exception as e:
        app.logger.error(f"创建目录失败: {str(e)}")
        return jsonify({'success': False, 'error': f'创建目录失败: {str(e)}'}), 500

# 删除目录 API
@app.route('/api/delete-directory', methods=['DELETE']) # 绑定删除文件夹接口
@login_required
def api_delete_directory():
    """删除目录 API"""
    try:
        data = request.get_json()  # 获取JSON数据
        if not data:
            return jsonify({'success': False, 'error': '无效的请求数据'}), 400
        
        dir_path = data.get('path', '').strip() # 获取要删的文件夹路径
        
        # 验证路径
        if not dir_path:
            return jsonify({'success': False, 'error': '目录路径不能为空'}), 400
        
        # 确保路径以 / 结尾
        if not dir_path.endswith('/'):
            dir_path += '/'
        
        app.logger.info(f"删除目录: {dir_path}")
        
        # 检查目录是否存在
        try:
            minio_client.stat_object(MINIO_BUCKET, dir_path) # 查存不存在
            # ← 成功=存在，抛异常=不存在
            # MinIO SDK 选择 异常机制 是因为：
            #       - 成功是常态（大多数调用都成功），用异常处理成功情况浪费
        except Exception:
            return jsonify({'success': False, 'error': '目录不存在'}), 404
        
        # 检查目录是否为空（不包含任何文件或子目录）
        objects = list(minio_client.list_objects(MINIO_BUCKET, prefix=dir_path, recursive=True)) # 把这个路径下所有的东西（包括子文件夹里的）都列出来
        
        # 过滤掉目录本身
        content_objects = [obj for obj in objects if obj.object_name != dir_path]
        
        if content_objects: # 如果过滤完还有东西
            return jsonify({'success': False, 'error': '目录不为空，无法删除'}), 400
        
        # 删除目录
        minio_client.remove_object(MINIO_BUCKET, dir_path)
        
        app.logger.info(f"目录删除成功: {dir_path}")
        return jsonify({'success': True, 'message': '目录删除成功'})
        
    except Exception as e:
        app.logger.error(f"删除目录失败: {str(e)}")
        return jsonify({'success': False, 'error': f'删除目录失败: {str(e)}'}), 500

# 上传文件 API
@app.route('/api/upload', methods=['POST']) # 绑定上传文件接口
@login_required
def api_upload_files():
    """文件上传 API"""
    try:
        # 获取上传路径
        upload_path = request.form.get('path', '') # 从表单里获取要传到哪个文件夹下
        
        # 检查是否有文件
        if 'files' not in request.files: # 如果前端没传叫files的字段
            return jsonify({'error': '没有选择文件'}), 400
        
        files = request.files.getlist('files')  # 把前端传来的多个文件拿出来变成列表
        if not files or all(f.filename == '' for f in files): # 如果列表为空，或者所有的文件名都是空的
            return jsonify({'error': '没有选择有效的文件'}), 400
        
        app.logger.info(f"开始上传 {len(files)} 个文件到路径: {upload_path}")
        
        uploaded_files = [] # 准备装上传成功的文件信息
        failed_files = []   # 准备装上传失败的文件信息
        
        for file in files:
            if file.filename == '': # 遍历到的文件名为空则跳过
                continue
                
            try:
                # 构建完整的对象路径
                if upload_path: # 如果有上传路径，目标文件夹
                    object_name = f"{upload_path}/{file.filename}" # 拼成 "文件夹/文件名"
                else: # 如果没目标文件夹（传到根目录）
                    object_name = file.filename
                
                # 检查文件是否已存在
                try:
                    minio_client.stat_object(MINIO_BUCKET, object_name) 
                    failed_files.append({ # 如果存在则塞进失败列表
                        'filename': file.filename,
                        'error': '文件已存在'
                    })
                    continue # 跳过不传文件
                except:
                    # 文件不存在，可以上传
                    pass
                
                # 上传文件到 MinIO
                file.seek(0)  # 重置文件指针，把文件的读取指针拨回最开头（防止之前有读取过导致指针在后面）
                file_data = file.read() # 把文件数据全读到内存里
                file_size = len(file_data)
                
                # 检查文件大小限制 (500MB)
                max_size = 500 * 1024 * 1024
                if file_size > max_size:
                    failed_files.append({
                        'filename': file.filename,
                        'error': f'文件过大 ({file_size / 1024 / 1024:.2f} MB > 500 MB)'
                    })
                    continue
                
                # 重置文件指针并上传
                file.seek(0) # 因为前面read过一次，指针又到末尾了，必须再拨回开头才能传
                minio_client.put_object( # 调用MinIO上传接口
                    MINIO_BUCKET,   # 桶名
                    object_name,    # 存的路径名
                    file,           # 文件流对象
                    file_size,
                    content_type=file.content_type or 'application/octet-stream'  # 文件类型，不知道就当二进制
                )
                
                uploaded_files.append({ # 上传成功，塞进成功列表
                    'filename': file.filename,
                    'object_name': object_name,
                    'size': file_size
                })
                
                app.logger.info(f"文件上传成功: {object_name}")
                
            except Exception as e:
                app.logger.error(f"上传文件 {file.filename} 失败: {str(e)}")
                failed_files.append({
                    'filename': file.filename,
                    'error': str(e)
                })
        
        # 构造返回结果
        result = {
            'success': len(uploaded_files) > 0,         # 只要有成功上传的就算整体成功
            'uploaded_count': len(uploaded_files),      # 成功上传的文件数量
            'failed_count': len(failed_files),          # 失败上传的文件数量
            'uploaded_files': uploaded_files,           # 成功的详细列表
            'failed_files': failed_files,               # 失败的详细列表
            'message': f'成功上传 {len(uploaded_files)} 个文件'
        }
        
        if failed_files:
            result['message'] += f'，{len(failed_files)} 个文件上传失败' # 添加上传失败的提示语
        
        return jsonify(result) # 返回JSON给前端
        
    except Exception as e:
        app.logger.error(f"文件上传处理失败: {str(e)}")
        return jsonify({'error': f'文件上传处理失败: {str(e)}'}), 500



if __name__ == '__main__':
    
    # 获取主机和端口配置
    host = os.getenv('FLASK_HOST')       # 从环境变量读取要运行在哪个IP上
    port = int(os.getenv('FLASK_PORT'))  # 读取端口号并转成整数


    print("=" * 60)
    print("Mildoc 管理后台启动中...")
    print(f"访问端口: {port}")
    print("=" * 60)
    
    app.run(
        host=host,
        port=port,
        debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true', # 看看要不要开启调试模式（代码改了自动重启）
        threaded=True,  # 开启多线程，能同时处理好几个人的请求
        processes=1     # 只开1个进程
    ) 