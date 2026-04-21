"""
企业微信API调用模块
支持客服界面消息的读取和发送功能

"""

import time
import logging
import requests
from typing import Dict, Optional # Optional表示这个变量可能是某个类型，也可能是None（空）
from config import Config

logger = logging.getLogger(__name__)

class WeComAPI:
    """企业微信API调用类"""
    
    def __init__(self):
        self.corp_id = Config.CORP_ID       # 企业微信企业ID（从配置读取）
        self.app_secret = Config.APP_SECRET # 应用密钥（从配置读取）
        self.base_url = "https://qyapi.weixin.qq.com/cgi-bin"  # 企业微信API基础地址
        self._kf_access_token = None        # 私有属性：客服access_token（缓存用）
        self._kf_token_expires_at = 0       # 私有属性：token过期时间戳（避免重复请求）
    
    
    def get_kf_access_token(self) -> Optional[str]:
        """获取客服专用access_token"""
        # 检查token是否过期。如果通行证存在 且 当前时间小于过期时间   **token缓存，防止每次发消息都去请求一次token**
        if self._kf_access_token and time.time() < self._kf_token_expires_at:
            return self._kf_access_token # 说明通行证还没过期，直接返回通行证，不用再去请求服务器（省时间防封禁）
        
        # 配置客服密钥
        secret = self.app_secret
        if not secret:
            logger.error("缺少APP_SECRET配置，无法获取客服access_token")
            return None # 代表失败
        
        # 无有效token则调用企业微信API获取
        url = f"{self.base_url}/gettoken"
        params = { # 发给服务器的参数
            'corpid': self.corp_id, # 企业ID
            'corpsecret': secret    # 应用密钥
        }
        
        try:
            response = requests.get(url, params=params, timeout=10) # 向企业微信服务器发送GET请求
            response.raise_for_status()                             # 主动抛出HTTP异常（如404/500）
            data = response.json()                                  # 把服务器返回的JSON格式字符串，自动转换成Python的字典
            
            if data.get('errcode') == 0: # 检查字典里的错误码，0代表企业微信那边处理成功
                # 缓存token + 计算过期事件（提前5分钟过期，避免临界值失效）
                self._kf_access_token = data.get('access_token')
                expires_in = data.get('expires_in', 7200) # 有效时长（秒），如果服务器没返回这个字段，默认就用7200秒（2小时）
                # 计算绝对过期时间戳：当前时间 + 有效期 - 300秒（提前5分钟判定为过期，防止用到最后一秒突然失效引发bug）
                self._kf_token_expires_at = time.time() + expires_in - 300  # 提前5分钟过期
                logger.info(f"获取客服access_token成功，有效期: {expires_in}秒")
                return self._kf_access_token
            else:
                logger.error(f"获取客服access_token失败: {data.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"获取客服access_token异常: {e}")
            return None
    
    def sync_kf_messages(self, token: str, open_kfid: str = "", cursor: str = "", limit: int = 1000) -> Optional[Dict]:
        """
        读取客服消息
        
        Args:
            token: 回调事件返回的token字段
            open_kfid: 指定拉取某个客服账号的消息
            cursor: 上一次调用时返回的next_cursor
            limit: 期望请求的数据量，默认1000
            
        Returns:
            消息列表数据
        """
        # 先获取通行证
        access_token = self.get_kf_access_token()
        if not access_token:
            return None
        
        url = f"{self.base_url}/kf/sync_msg"    # 拼接读取消息的接口网址
        params = {'access_token': access_token} # 把通行证放到URL的参数里（?access_token=xxx这种形式）
        
        # 准备POST请求要发送的报文主体（字典格式）
        data = {
            'token': token, # 回调给的token（企业微信用来定位是哪次对话的消息）
            'limit': limit, # 期望拉取的消息条数
            'voice_format': 0 # 语音消息的格式，0代表Amr格式，1-Silk
        }
        
        if cursor: # 如果传入了游标（翻页标记）
            data['cursor'] = cursor
        if open_kfid:
            data['open_kfid'] = open_kfid # 把客服账号ID加到要发送的数据里
        
        try:
            # 发送POST请求，json=data会自动把字典转成JSON格式发过去
            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errcode') == 0: # 判断企业微信是否处理成功
                logger.info(f"读取客服消息成功，获取到 {len(result.get('msg_list', []))} 条消息")
                return result
            else:
                logger.error(f"读取客服消息失败: {result.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"读取客服消息异常: {e}")
            return None
    
    def send_kf_message(self, touser: str, open_kfid: str, msgtype: str, content: Dict, msgid: str = None) -> Optional[Dict]:
        """
        发送客服消息(通用，类内部使用)
        
        Args:
            touser: 接收消息的客户UserID
            open_kfid: 发送消息的客服帐号ID
            msgtype: 消息类型 (text, image, voice, video, file, link, miniprogram, msgmenu, location)
            content: 消息内容
            msgid: 指定消息ID
            
        Returns:
            发送结果
        """
        access_token = self.get_kf_access_token()
        if not access_token:
            return None
        
        url = f"{self.base_url}/kf/send_msg"     # 拼接发送消息的网址
        params = {'access_token': access_token}
        
        data = { # 报文主体
            'touser': touser,       # 接收消息的客户UserID
            'open_kfid': open_kfid, # 发送消息的客服帐号ID
            'msgtype': msgtype      # 消息类型
        }
        
        # 添加消息内容
        data[msgtype] = content
        
        if msgid: # 如果调用者指定了消息ID（一般用于防止重复发送）
            data['msgid'] = msgid
        
        try:
            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errcode') == 0: # 判断是否发送成功
                logger.info(f"发送客服消息成功，msgid: {result.get('msgid')}")
                return result
            else:
                logger.error(f"发送客服消息失败: {result.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"发送客服消息异常: {e}")
            return None
    
    def send_kf_text_message(self, touser: str, open_kfid: str, content: str, msgid: str = None) -> Optional[Dict]:
        """发送文本消息"""
        text_content = {'content': content}
        return self.send_kf_message(touser, open_kfid, 'text', text_content, msgid) # 直接调用上面的通用方法，指定类型为'text'
    
    def send_kf_image_message(self, touser: str, open_kfid: str, media_id: str, msgid: str = None) -> Optional[Dict]:
        """发送图片消息"""
        image_content = {'media_id': media_id}
        return self.send_kf_message(touser, open_kfid, 'image', image_content, msgid)
    
    def send_kf_link_message(self, touser: str, open_kfid: str, title: str, desc: str, url: str, thumb_media_id: str, msgid: str = None) -> Optional[Dict]:
        """发送链接消息"""
        link_content = {
            'title': title, # 链接标题
            'desc': desc, # 链接描述
            'url': url, # 链接URL
            'thumb_media_id': thumb_media_id # 链接缩略图媒体ID
        }
        return self.send_kf_message(touser, open_kfid, 'link', link_content, msgid)
        
    # 定义一个专门在"特定事件"发生时发消息的方法
    def send_event_response_message(self, code: str, content: str) -> Optional[Dict]:
        """
        发送事件响应消息（如欢迎语、结束语）
        
        Args:
            code: 事件响应码
            msgtype: 消息类型 (text, msgmenu)
            content: 消息内容
            
        Returns:
            发送结果
        """
        access_token = self.get_kf_access_token()
        if not access_token:
            return None
        
        url = f"{self.base_url}/kf/send_msg_on_event"
        params = {'access_token': access_token}
        
        data = {
            'code': code,               # 用户进入聊天时，企业微信推送过来的一个事件码，拿着这个码才能发欢迎语
            'msgtype': 'text',          # 消息类型
            'text': {'content': content} # 文本消息内容
        }
        
        try:
            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errcode') == 0: # 判断是否发送成功
                logger.info(f"发送事件响应消息成功，code: {code}")
                return result
            else:
                logger.error(f"发送事件响应消息失败: {result.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"发送事件响应消息异常: {e}")
            return None
    
    # 定义一个去查当前用户"接待状态"的方法
    def get_service_session_state(self, external_userid: str, open_kfid: str) -> Optional[Dict]:
        """
        获取会话状态
        
        Args:
            open_kfid: 客服帐号ID
            external_userid: 用户的external_userid
            
        Returns:
            会话状态数据，包含以下字段:
            - service_state: 会话状态 (0: 未接待, 1: 由智能助手接待, 2: 接待池等待中, 3: 人工接待, 4: 用户已确认接待结束)
            - service_userid: 接待客服的userid (当 service_state 为3时返回)
            - service_session_id: 会话ID
        """
        access_token = self.get_kf_access_token()
        if not access_token:
            return None
        
        url = f"{self.base_url}/kf/service_state/get"
        params = {'access_token': access_token}
        
        data = {    # 准备报文
            'open_kfid': open_kfid,     # 客服帐号ID
            'external_userid': external_userid # 用户的external_userid
        }
        
        try:
            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errcode') == 0: # 判断是否获取成功
                logger.info(f"获取会话状态成功，状态: {result.get('service_state')}")
                return result
            else:
                logger.error(f"获取会话状态失败: {result.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"获取会话状态异常: {e}")
            return None
    
    # 定义一个"改变接待状态"的方法（比如转人工）
    def update_service_session_state(self, external_userid: str, open_kfid: str, service_state: int, service_userid: str = None) -> Optional[Dict]:
        """
        变更会话状态
        
        Args:
            open_kfid: 客服帐号ID
            external_userid: 用户的external_userid
            service_state: 要变更的会话状态 (0: 结束接待, 1: 开启由智能助手接待, 2: 进入接待池由人工接待)
            service_userid: 接待客服的userid (当service_state为3时必填)
            
        Returns:
            变更结果
        """
        access_token = self.get_kf_access_token()
        if not access_token:
            return None
        
        url = f"{self.base_url}/kf/service_state/trans"
        params = {'access_token': access_token}
        
        data = {    # 准备报文
            'open_kfid': open_kfid,     # 哪个客服账号
            'external_userid': external_userid, # 哪个用户
            'service_state': service_state # 要变成什么状态
        }
        
        # 当状态为3（开启由人工接待）时，必须指定接待人。切入状态3（转人工），企业微信的后台必须明确知道是“张三”还是“李四”接的单，否则后续的绩效统计、聊天记录归档全都会乱套。
        if service_state == 3:
            if not service_userid: # 但是没有传真人客服的ID
                logger.error("变更会话状态失败: service_state为3时必须指定service_userid")
                return None
            data['service_userid'] = service_userid # 如果传了真人客服ID，就加到报文里
        
        try:

            logger.info(f"变更会话状态请求: {data}")
            logger.info(f"变更会话状态请求: {url}")
            logger.info(f"变更会话状态请求: {params}")

            response = requests.post(url, params=params, json=data, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            if result.get('errcode') == 0: # 判断是否变更成功
                logger.info(f"变更会话状态成功，状态: {service_state}")
                return result
            else:
                logger.error(f"变更会话状态失败: {result.get('errmsg')}")
                return None
                
        except Exception as e:
            logger.error(f"变更会话状态异常: {e}")
            return None
    


# 全局API实例
wecom_api = WeComAPI() 