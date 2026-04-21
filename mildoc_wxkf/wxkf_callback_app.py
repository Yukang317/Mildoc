"""
企业微信回调服务端
功能：搭建Web服务，接收企业微信的回调验证（GET）和消息推送（POST），完成官方要求的解密验签，提取有用信息送给后台。

"""

import os
import logging
import json
from urllib.parse import unquote # 导入URL解码工具，因为微信发来的有些参数是被编码过的（比如空格变成%20），需要翻译回来
from flask import Flask, request, abort # 从Flask框架导入核心类：Flask(网站本体)、request(来访者的请求数据包)、abort(直接把来访者踢走报错)
from WXBizMsgCrypt import WXBizMsgCrypt # 导入企业微信官方提供的“加解密翻译官”第三方库，专门对付微信那套奇葩加密
from config import Config

# 配置日志
logging.basicConfig(
    level=logging.INFO, # 只打印INFO级别及以上的日志（不打印DEBUG那些废话）
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' # 时间 - 文件名 - 级别 - 内容
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__) # 实例化一个Flask网站应用，__name__帮它找到自己所在的文件夹

def get_wecom_config():
    """获取企业微信配置，优先从环境变量获取"""
    corp_id = Config.CORP_ID                    # 企业 ID
    token = Config.TOKEN
    encoding_aes_key = Config.ENCODING_AES_KEY  # AES密钥
    
    if not token:
        logger.error("缺少TOKEN配置，请设置环境变量TOKEN或修改代码中的TOKEN变量")
        return None, None, None
    
    if not encoding_aes_key:
        logger.error("缺少ENCODING_AES_KEY配置，请设置环境变量WECOM_ENCODING_AES_KEY或修改代码中的ENCODING_AES_KEY变量")
        return None, None, None
    
    return corp_id, token, encoding_aes_key


@app.before_request # Flask“钩子”装饰器：“在处理任何一个具体请求之前，必须先自动执行下面这个函数”
def log_request_info():
    """记录请求信息"""
    # 记录请求的基本信息
    logger.info(f"=== 收到HTTP请求 ===")
    logger.info(f"请求方法: {request.method}")  # GET / POST
    logger.info(f"请求URL: {request.url}")      # 完整网址
    logger.info(f"请求路径: {request.path}")    # 路径部分，不包含查询参数
    logger.info(f"客户端IP: {request.remote_addr}") # 哪个IP发的请求？
    
    # 记录请求头
    headers = dict(request.headers)
    logger.info(f"请求头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
    
    # 记录查询参数
    if request.args: # 如果网址后面带了问号和参数（比如 ?name=abc）
        args = dict(request.args) # 把参数转换为字典
        logger.info(f"查询参数: {json.dumps(args, indent=2, ensure_ascii=False)}")
    
    # 记录请求体（仅对POST/PUT等方法），POST带大数据包
    if request.method in ['POST', 'PUT', 'PATCH']:
        try:
            # 如果发的是JSON格式的数据
            if request.content_type and 'application/json' in request.content_type:
                # JSON数据
                json_data = request.get_json() # 直接把JSON解析成Python字典
                if json_data:
                    logger.info(f"请求JSON: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            
            # 如果不是JSON格式（微信发的是XML，就走这个分支）
            else:
                # 原始数据，把最原始的请求体当成纯文本拿出来
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    logger.info(f"请求体: {raw_data[:500]}...")  # 限制长度避免日志过长
        except Exception as e:
            logger.warning(f"读取请求体时出错: {e}")

@app.after_request # Flask“钩子”装饰器：“在处理完请求、准备把结果发回给来访者之前，自动执行下面这个函数”
def log_response_info(response):
    """记录响应信息"""
    logger.info(f"=== HTTP响应 ===")
    logger.info(f"响应状态码: {response.status_code}")
    logger.info(f"响应状态: {response.status}")
    
    # 记录响应头
    headers = dict(response.headers)
    logger.info(f"响应头: {json.dumps(headers, indent=2, ensure_ascii=False)}")
    
    # 记录响应内容
    try:
        if response.content_type and 'application/json' in response.content_type:
            # JSON响应
            logger.info(f"响应JSON: {response.get_data(as_text=True)}") # 打印JSON内容
        else:
            # 其他类型响应（比如XML等）
            response_data = response.get_data(as_text=True)
            if response_data:
                if len(response_data) > 500:
                    logger.info(f"响应内容: {response_data[:500]}...")
                else:
                    logger.info(f"响应内容: {response_data}")
            else:
                logger.info("响应内容: 无内容")
    except Exception as e:
        logger.warning(f"读取响应内容时出错: {e}")
    
    logger.info(f"=== 请求处理完成 ===\n")
    return response


def get_wxcrypt():
    """
    获取企业微信加解密工具对象
    """
    corp_id, token, encoding_aes_key = get_wecom_config()
    if not all([corp_id, token, encoding_aes_key]): # 三个里面任何一个为空
        logger.error("企业微信配置不完整")
        abort(500) # 直接中断，给来访者返回500服务器内部错误
    # 把三个法宝喂给官方库，造出一个“翻译官”对象并返回
    return WXBizMsgCrypt(token, encoding_aes_key, corp_id)


@app.route('/callback/command', methods=['GET']) # 路由装饰器：GET方式访问 网址/callback/command，执行下面的函数
def wecom_callback_get(): # 处理GET请求的函数（专门用来验证服务器有效性）
    """
    企业微信回调接口
    GET请求：用于验证回调URL的有效性
    """
    # 获取加解密工具对象
    wxcrypt = get_wxcrypt()
    
    # 获取URL参数
    msg_signature = request.args.get('msg_signature', '') # 从网址参数里拿“签名”
    timestamp = request.args.get('timestamp', '') # 从网址参数里拿“时间戳”
    nonce = request.args.get('nonce', '') # 从网址参数里拿“随机数”
    
    logger.info(f"收到回调请求 - Method: {request.method}, msg_signature: {msg_signature}, timestamp: {timestamp}, nonce: {nonce}")
    
    # URL验证
    echostr = request.args.get('echostr', '') # 验证码字符串，微信发来的随即乱码
    if not echostr:
        logger.error("GET请求缺少echostr参数")
        abort(400)
    
    if not all([msg_signature, timestamp, nonce]):
        logger.error("GET请求缺少必要参数")
        abort(400)
    
    try:
        
        # URL解码echostr参数
        echostr = unquote(echostr) # 把验证码里被编码的特殊字符还原（比如%3D变成=）
        logger.info(f"开始验证URL - echostr: {echostr[:50]}...")
        
        logger.info(f"echostr: {echostr}")

        # 验证URL并解密echostr
        result = wxcrypt.VerifyURL(msg_signature, timestamp, nonce, echostr)
        
        logger.info(f"验证URL结果: {result}")

        # 检查返回值类型
        if isinstance(result, tuple):  # 官方库有时返回一个元组(状态码, 解密结果)，有时只返回状态码
            ret, reply_echostr = result
        else: # 返回不是元组
            ret = result # 那就是状态码
            reply_echostr = None  # 解密结果为空
        
        logger.info(f"验证结果 - 返回码: {ret}")
        if ret != 0:
            logger.error(f"URL验证失败，错误码: {ret}")
            if ret == -40001:
                logger.error("签名验证失败 - 请检查Token配置是否与企业微信后台一致")
            elif ret == -40002:
                logger.error("AES解密失败或CorpID不匹配 - 请检查EncodingAESKey和CorpID配置")
            else:
                logger.error(f"未知错误码: {ret}")
            abort(403)
        
        logger.info("URL验证成功")
        return reply_echostr
    except Exception as e:
        logger.error(f"URL验证过程发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}") # 打印出具体在哪一行炸的
        abort(500)
        
    



@app.route('/callback/command', methods=['POST']) # 路由：如果有人用POST方式访问 /callback/command，执行下面函数
def wecom_callback_post(): # 处理POST请求的函数（专门用来接收微信真正推送的用户消息）
    """
    企业微信回调接口
    POST请求：接收企业微信推送的消息和事件
    """
    # 获取加解密工具对象
    wxcrypt = get_wxcrypt()
    
    # 获取URL参数
    msg_signature = request.args.get('msg_signature', '')
    timestamp = request.args.get('timestamp', '')
    nonce = request.args.get('nonce', '')
    
    logger.info(f"收到回调请求 - Method: {request.method}, msg_signature: {msg_signature}, timestamp: {timestamp}, nonce: {nonce}")
    
    # 接收消息
    if not all([msg_signature, timestamp, nonce]):
        logger.error("POST请求缺少必要参数")
        abort(400)
        
    try:
        # 获取POST数据
        post_data = request.get_data(as_text=True)  # 把微信POST过来的加密信封（一坨乱码XML）拿出来
        if not post_data:
            logger.error("POST请求体为空")
            abort(400)
        
        logger.info(f"收到POST消息: {post_data[:200]}...")
        
        # 解密消息
        ret, msg = wxcrypt.DecryptMsg(post_data, msg_signature, timestamp, nonce)
        
        if ret != 0:
            logger.error(f"消息解密失败，错误码: {ret}")
            abort(403)
        
        logger.info(f"消息解密成功: {msg}") # 打印解密后的明文XML
        
        # 处理消息（这里可以根据业务需求进行扩展）
        response_msg = handle_message(msg) # 把明文XML丢给专门处理业务的函数，看它要不要给用户回话。返回str
        
        if response_msg:
            # 加密响应消息
            ret, encrypted_msg = wxcrypt.EncryptMsg(response_msg, nonce, timestamp) # “翻译官”把回复用AES锁上，盖上签名
            if ret == 0:
                logger.info("响应消息加密成功")
                return encrypted_msg # 把加密后的信封丢给微信
            else:
                logger.error(f"响应消息加密失败，错误码: {ret}")
        
        # 返回空字符串表示成功接收但不回复，否则微信会一直重试发这条消息
        return ''
        
    except Exception as e:
        logger.error(f"处理POST请求时发生错误: {str(e)}")
        abort(500)
    

def handle_message(msg): # 专门处理解密后XML消息的函数
    """
    处理解密后的消息
    
    Args:
        msg: 解密后的XML消息
        
    Returns:
        str: 要回复的消息（XML格式），如果不需要回复则返回None
    """
    try:
        import xml.etree.ElementTree as ET # 导入Python自带的XML解析器，把XML格式的字符串当成树来拆解
        import time
        
        # 解析XML消息
        root = ET.fromstring(msg) # 把XML字符串变成一棵“树”，root就是树的根节点
        msg_type = root.find('MsgType').text if root.find('MsgType') is not None else '' # 在树里找叫MsgType的标签，拿出里面的文字（比如event代表事件，text代表文字）
        msg_id = root.find('MsgId').text if root.find('MsgId') is not None else '' # 找MsgId标签拿消息ID
        
        logger.info(f"处理消息 - 类型: {msg_type}, 消息ID: {msg_id}")
        
        # 根据消息类型进行处理
        if msg_type == 'event': # 如果发现这是一个“事件”（比如用户进入会话、离开等系统动作，不是具体消息）
            # 处理事件消息
            event = root.find('Event').text if root.find('Event') is not None else '' # 找Event标签看具体是啥事件
            logger.info(f"收到事件: {event}")
            
            return process_event_message(event, root)
                
        # 对于其他类型的消息，记录日志但不回复
        logger.info(f"收到其他类型消息，暂不处理: {msg_type}")
        return None
        
    except Exception as e:
        logger.error(f"处理消息时发生错误: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        return None



def process_event_message(event, root):
    """
    处理事件类型的消息
    
    Args:
        event: 事件类型
        from_user: 发送者
        to_user: 接收者
        root: XML根节点
        
    Returns:
        str: 回复消息，如果不需要回复则返回None
    """
    if event == 'kf_msg_or_event': # 如果事件名字叫 'kf_msg_or_event'（这是微信客服特有的，代表有客服消息或事件）
        # 微信客服消息或事件
        logger.info("收到微信客服事件，开始处理客服消息")
        
        # 获取Token和OpenKfId
        token = root.find('Token').text if root.find('Token') is not None else '' # 从XML树里掏出Token（临时凭证）
        open_kfid = root.find('OpenKfId').text if root.find('OpenKfId') is not None else ''
        
        if token and open_kfid:
            # 异步处理客服消息（避免阻塞回调响应），为什么要开线程 (threading.Thread)**：企业微信有一个**死规定：你的接口必须在 5 秒内给它返回内容！**

            import threading # 导入多线程模块
            
            def process_kf_messages(): # 闭包
                try:
                    # 延迟导入避免循环导入，防止死锁
                    from kf_message_handler import KfMessageHandler # 导入消息处理器
                    kf_handler = KfMessageHandler()
                    kf_handler.process_kf_event(token, open_kfid)
                except Exception as e:
                    logger.error(f"处理客服消息异常: {e}")
            
            # 启动后台线程处理
            thread = threading.Thread(target=process_kf_messages) # 创建后台线程，把闭包函数放入
            thread.daemon = True # 把线程设为“守护线程”（意思是如果主程序退出了，这个线程也会被强制杀掉，不会变成孤魂野鬼）
            thread.start()
            
            logger.info(f"已启动客服消息处理线程 - OpenKfId: {open_kfid}")
        else:
            logger.error("客服事件缺少必要参数 - Token或OpenKfId为空")
        
        # 客服事件不需要回复
        return None
    
    # 其他事件类型
    logger.info(f"收到其他事件类型: {event}")
    return None

@app.route('/health', methods=['GET'])
def health_check(): # 运维人员或者负载均衡器会定期来 ping 这个网址
    """健康检查接口"""
    return {'status': 'ok', 'message': '企业微信回调服务运行正常'}

@app.route('/', methods=['GET']) # 路由：访问根目录 网址/
def index():
    return '''
    <h1>企业微信回调服务</h1>
    <p>服务正在运行...</p>
    '''



if __name__ == '__main__':
    # 检查配置
    corp_id, token, encoding_aes_key = get_wecom_config()
    
    if not all([corp_id, token, encoding_aes_key]):
        logger.error("配置不完整，请检查企业微信相关配置")
        logger.info("请设置以下环境变量或修改代码中的配置:")
        logger.info("- WECOM_CORP_ID: 企业ID")
        logger.info("- WECOM_TOKEN: 应用Token")
        logger.info("- WECOM_ENCODING_AES_KEY: 应用EncodingAESKey")
        exit(1)
    
    logger.info("企业微信回调服务启动中...")
    logger.info(f"企业ID: {corp_id}")
    logger.info(f"Token: {token[:10]}..." if token else "Token: 未配置")
    logger.info(f"EncodingAESKey: {encoding_aes_key[:10]}..." if encoding_aes_key else "EncodingAESKey: 未配置")

    # 启动服务
    port = Config.PORT
    logger.info(f"启动服务 - 端口: {port}")
    host = Config.HOST
    logger.info(f"启动服务 - 主机: {host}")
    debug = Config.DEBUG
    logger.info(f"启动服务 - 调试模式: {debug}")

    app.run(host=host, port=port, debug=debug, threaded=True, processes=1)  # 真正让Flask网站跑起来！threaded=True表示支持多线程并发访问