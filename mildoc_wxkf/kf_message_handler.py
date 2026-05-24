import logging
import time
from typing import Dict 
from wecom_api import wecom_api             # 导入企业微信API封装模块，里面封装了所有跟微信服务器通信的方法（发消息、拉消息等）
from config import Config
from cursor_manager import cursor_manager   # 导入游标管理器，用来记录"上次拉消息拉到哪里了"，防止重复拉取
from rag_service import get_rag_service
logger = logging.getLogger(__name__)        # 给当前文件创建一个专属的日志记录器，打印时会带上文件名


WELCOME_MESSAGE = '''🎉 欢迎使用微信客服！

我是您的专属客服助手，很高兴为您服务！

🔹 有任何问题随时咨询
🔹 输入"帮助"查看功能菜单
🔹 我们提供7×24小时服务

请问有什么可以帮助您的吗？'''



 # 定义客服消息处理器的主类，所有的消息处理逻辑都被装在这里面
class KfMessageHandler:
    """微信客服消息处理器"""

    def __init__(self):
        self.processed_messages = set()  # 暂存内存中的消息（已处理过的消息），用来去重缓存
    
    # 主入口方法
    def process_kf_event(self, token: str, open_kfid: str) -> bool:
        """
        处理客服事件，拉取并处理消息（优化版本）
        
        Args:
            token: 回调事件中的token
            open_kfid: 客服账号ID
            
        Returns:
            处理是否成功
        """
        try:
            # 从持久化存储获取该客服账号的最后cursor
            cursor = cursor_manager.get_cursor(open_kfid) # 从数据库/文件里读取这个客服账号上次拉消息拉到哪个位置了（cursor就是一个位置标记）
            
            # 确定拉取限制：如果没有cursor（首次或丢失），只拉取最近1条消息作为保底。正常拉100条
            limit = 1 if not cursor else 100
            
            logger.info(f"开始拉取客服消息 - OpenKfId: {open_kfid}, Cursor: {'有' if cursor else '无'}, Limit: {limit}")
            
            # 拉取消息，这个cursor是我传入旧的，微信会返回新的cursor给我
            result = wecom_api.sync_kf_messages(token, open_kfid, cursor, limit)
            if not result:
                logger.error("拉取客服消息失败")
                return False
            
            msg_list = result.get('msg_list', [])           # 消息列表
            next_cursor = result.get('next_cursor', '')     # 下一次的cursor（微信给的“下一页页码”）
            has_more = result.get('has_more', 0)           # 是否还有更多消息。0 表示拉完了
            
            logger.info(f"拉取到 {len(msg_list)} 条客服消息，next_cursor: {'有' if next_cursor else '无'}, has_more: {has_more}")
            
            # 处理每条消息（带去重）
            processed_count = 0 # 本轮处理了多少条消息
            new_messages = 0    # 记录本轮有多少条是真正的新消息（排除了重复的）
            
            # 遍历拉回来的每一条消息（msg是一个字典）
            for msg in msg_list:
                msgid = msg.get('msgid', '')    # 取出这条消息的ID
                
                # 检查消息是否已处理（双重检查：内存缓存 + 数据库）
                if msgid in self.processed_messages or cursor_manager.is_message_processed(msgid): # 双重去重：先查内存里的集合，再去数据库查一遍
                    logger.debug(f"消息已处理，跳过: {msgid}")
                    continue
                # - `or` 运算符是"短路求值"。先查内存（快），如果内存里命中了就直接跳过，根本不会去碰数据库。只有内存里没有时，才会慢悠悠地去查数据库。这就实现了"快速通道 + 安全兜底"的完美配合。

                
                # 处理新消息
                if self.process_single_kf_message(msg): # 调用专门处理单条消息的方法，如果处理成功返回True
                    processed_count += 1
                    new_messages += 1
                    # 添加到内存缓存
                    self.processed_messages.add(msgid)
                    
                    # 限制内存缓存大小
                    if len(self.processed_messages) > 1000:
                        # 清理一半的缓存
                        self.processed_messages = set(list(self.processed_messages)[500:])
            
            # 保存cursor到持久化存储
            if next_cursor: # 如果微信给了一个新的cursor（说明拉到了东西）
                cursor_manager.save_cursor(open_kfid, next_cursor, new_messages) # 把新cursor存到数据库里，下次从这里开始拉
                logger.info(f"已保存新cursor - OpenKfId: {open_kfid}, 新消息数: {new_messages}")
            
            # 如果还有更多消息，继续拉取（但要避免无限循环，递归调用）  
            #       - 因为每次递归时微信返回的 `next_cursor` 会自动更新，不需要你手动维护循环变量。代码更简洁。而且有 `new_messages > 0` 
            #         这个安全阀——如果连续拉到的全是重复消息（新消息为0），就不再递归了，避免死循环。
            if has_more == 1 and new_messages > 0: # 如果积压了10万条消息会报错
                logger.info("还有更多消息，继续拉取...")
                return self.process_kf_event(token, open_kfid)
            
            logger.info(f"客服消息处理完成 - 总拉取: {len(msg_list)}, 新处理: {processed_count}")
            return True
            
        except Exception as e:
            logger.error(f"处理客服事件异常: {e}")
            return False
    
    def process_single_kf_message(self, msg: Dict) -> bool:
        """
        处理单条客服消息（优化版本）
        
        Args:
            msg: 消息数据
            
        Returns:
            处理是否成功
        """
        try:
            msgid = msg.get('msgid', '')                        # 消息唯一ID
            open_kfid = msg.get('open_kfid', '')                # 客服账号ID
            external_userid = msg.get('external_userid', '')    # 客户外部ID（微信用户的表示）
            send_time = msg.get('send_time', 0)                # 消息发送时间（Unix时间戳）
            origin = msg.get('origin', 0)                       # 消息来源类型：3-微信客户发送 4-系统推送事件 5-接待人员发送
            servicer_userid = msg.get('servicer_userid', '')    # 接待人员外部ID（微信用户的表示）
            msgtype = msg.get('msgtype', '')                    # 消息类型（如text、image等）
            
            logger.info(f"处理客服消息 - msgid: {msgid}, 类型: {msgtype}, 来源: {origin}, 客户: {external_userid}")
            
            # 检查消息时效性（仅对客户发送的消息进行时效检查）
            current_time = int(time.time())                 # 当前时间（Unix时间戳）
            message_age_seconds = current_time - send_time  # 消息年龄（秒）：现在-发送
            message_age_minutes = message_age_seconds / 60  # 消息年龄（分钟）
            
            # 标记消息为已处理，是否给用户发了回复，默认没发
            reply_sent = False
            
            # 只处理微信客户发送的消息
            if origin == 3 and external_userid:
                # 检查消息是否在10分钟内发送。超过十分钟客户可能离开
                if message_age_minutes <= 10:
                    reply_sent = self.handle_customer_message(msg) # 调用处理客户消息的方法
                    logger.info(f"消息已处理，消息年龄：{message_age_minutes:.1f}分钟")
                else:
                    logger.info(f"消息超过10分钟时效限制，不予回复 - msgid: {msgid}, "
                              f"消息时间: {send_time}, 当前时间: {current_time}, "
                              f"消息年龄: {message_age_minutes:.1f}分钟")
                    # 虽然不回复，但仍然标记为已处理，避免重复处理
                    reply_sent = True
            elif origin == 4:
                # 处理系统事件
                self.handle_system_event(msg)
            elif origin == 5:
                # 接待人员发送的消息，记录日志
                logger.info(f"接待人员 {servicer_userid} 发送消息: {msgtype}")
            
            # 标记消息为已处理
            cursor_manager.mark_message_processed(
                msgid, open_kfid, external_userid, msgtype, origin, reply_sent
            ) # 不管是客户消息还是系统事件，都去数据库里打个勾，说"这条处理过了"
            
            return True
            
        except Exception as e:
            logger.error(f"处理单条客服消息异常: {e}")
            return False
    
    

    
    def handle_customer_message(self, msg: Dict) -> bool:
        """
        处理客户发送的消息（origin==3）
        
        Returns:
            是否发送了回复
        """
        try:
            msgtype = msg.get('msgtype', '')                    # 消息类型
            external_userid = msg.get('external_userid', '')    # 客户外部ID（微信用户的表示）
            open_kfid = msg.get('open_kfid', '')                # 客服账号ID
                        
            reply_sent = False  # 默认没发回复

            service_state = self.get_service_session_state(external_userid, open_kfid) # 获取服务会话状态

            if (service_state != 1 and service_state != 0): # 如果不是智能助手接待和未分配
                # 如果不是智能助手接待状态，则不处理
                logger.info(f"非智能助手接待状态，不处理消息 - open_kfid: {open_kfid}, 客户: {external_userid}")
                return False
            
            if msgtype == 'text':
                # 处理文本消息，使用智能回复生成回复内容
                text_data = msg.get('text', {})
                content = text_data.get('content', '')
                
                logger.info(f"收到客户文本消息: {content}")
                
                if '转人工' in content:
                    # 转人工
                    reply_content = "🤖 收到您的转人工请求，我会尽快转接人工服务。"
                    reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content) # 发送客户回复
                    
                    self.update_service_session_state_to_service_pool(external_userid, open_kfid) # 更新服务会话状态为服务池，即“转人工排队”
                else:
                    # 生成智能回复
                    reply_content = self.get_smart_reply(content) # 智能回复，使用RAG服务获取回复内容，并记录token消耗情况和参考文档
                    # 发送回复
                    if reply_content:
                        reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content) # 发送客户回复
            
            elif msgtype in ['image', 'voice', 'video', 'file']:
                # 处理多媒体消息，暂不处理，仅作简单回复收到消息
                replies = {
                    'image': '''📷 收到您发送的图片，请简单描述一下图片内容，我能更好地为您服务！''',
                    
                    'voice': '''🎤 收到您的语音消息，感谢您的留言！''',
                    
                    'video': '''🎬 收到您发送的视频，感谢分享！''',
                    
                    'file': '''📎 收到您发送的文件，我会尽快查看处理。'''
                }
                reply_content = replies.get(msgtype, '收到您的消息，感谢分享！我会尽快为您处理。') # 默认回复
                if reply_content:
                    reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content) # 发送客户回复
            
            elif msgtype == 'location':
                # 处理位置消息，暂不处理，仅作简单回复收到消息
                location = msg.get('location', {})
                name = location.get('name', '')
                address = location.get('address', '')
                reply_content = f"📍 收到您分享的位置：{name}\n地址：{address}\n\n感谢分享！如需导航或周边服务，请告诉我具体需求。"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            elif msgtype == 'link':
                # 处理链接消息，暂不处理，仅作简单回复收到消息
                link = msg.get('link', {})
                title = link.get('title', '')
                reply_content = f"🔗 收到您分享的链接：{title}\n\n感谢分享！我会查看相关内容。"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            elif msgtype == 'business_card':
                # 处理名片消息，暂不处理，仅作简单回复收到消息
                reply_content = "👤 收到您的名片，感谢分享联系方式！\n\n如有业务合作需求，我们会及时与您联系。"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            elif msgtype == 'miniprogram':
                # 处理小程序消息，暂不处理，仅作简单回复收到消息
                mini = msg.get('miniprogram', {})
                title = mini.get('title', '')
                reply_content = f"📱 收到您分享的小程序：{title}\n\n感谢分享！我会查看相关功能。"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            elif msgtype == 'channels_shop_product':
                # 处理视频号商品消息，暂不处理，仅作简单回复收到消息
                product = msg.get('channels_shop_product', {})
                title = product.get('title', '')
                price = product.get('sales_price', '')
                reply_content = f"🛍️ 收到您关注的商品：{title}\n价格：{price}分\n\n如需了解更多商品信息或购买咨询，请告诉我！"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            elif msgtype == 'channels_shop_order':
                # 处理视频号订单消息，暂不处理，仅作简单回复收到消息
                order = msg.get('channels_shop_order', {})
                order_id = order.get('order_id', '')
                state = order.get('state', '')
                reply_content = f"📦 收到您的订单信息：{order_id}\n状态：{state}\n\n如需查询订单详情或有其他问题，请随时联系我！"
                reply_sent = self.send_kf_reply(external_userid, open_kfid, reply_content)
            
            return reply_sent
            
        except Exception as e:
            logger.error(f"处理客户消息异常: {e}")
            return False
    
    # 处理系统推送事件的方法（不返回值，因为系统事件不需要回复用户）
    def handle_system_event(self, msg: Dict) -> None:
        """处理系统事件"""
        try:
            event_data = msg.get('event', {})  # event部分（事件详情都塞在里面）
            event_type = event_data.get('event_type', '')   # 事件类型名称（字符串）
            
            logger.info(f"处理系统事件: {event_type}")
            
            if event_type == 'enter_session':
                # 用户进入会话事件
                self.handle_enter_session_event(event_data)
            elif event_type == 'msg_send_fail':
                # 消息发送失败事件
                self.handle_send_fail_event(event_data)
            elif event_type == 'servicer_status_change':
                # 接待人员状态变更事件
                self.handle_servicer_status_change_event(event_data)
            elif event_type == 'session_status_change':
                # 会话状态变更事件
                self.handle_session_change_event(event_data)
            elif event_type == 'user_recall_msg':
                # 用户撤回消息事件
                self.handle_user_recall_event(event_data)
            elif event_type == 'servicer_recall_msg':
                # 接待人员撤回消息事件
                self.handle_servicer_recall_event(event_data)
            
        except Exception as e:
            logger.error(f"处理系统事件异常: {e}")
    
    def handle_enter_session_event(self, event_data: Dict) -> None:
        """处理用户进入会话事件"""
        try:
            external_userid = event_data.get('external_userid', '')
            open_kfid = event_data.get('open_kfid', '')
            scene = event_data.get('scene', '')     # 进入场景
            welcome_code = event_data.get('welcome_code', '')
            
            logger.info(f"用户 {external_userid} 进入会话 - 场景: {scene} - 欢迎码: {welcome_code} - open_kfid: {open_kfid}")
            
            # 发送欢迎消息
            welcome_msg = WELCOME_MESSAGE
            # 使用事件响应消息接口发送欢迎语
            wecom_api.send_event_response_message(welcome_code, welcome_msg)
            
        except Exception as e:
            logger.error(f"处理进入会话事件异常: {e}")
    
    def handle_send_fail_event(self, event_data: Dict) -> None:
        """处理消息发送失败事件，暂不处理，仅做日志记录"""
        try:
            external_userid = event_data.get('external_userid', '')
            fail_msgid = event_data.get('fail_msgid', '')           # 失败消息的ID
            fail_type = event_data.get('fail_type', 0)              # 失败类型编号
            
            fail_reasons = {
                0: "未知原因",
                1: "客服账号已删除",
                2: "应用已关闭",
                4: "会话已过期，超过48小时",
                5: "会话已关闭",
                6: "超过5条限制",   # 客服主动发消息有数量限制
                7: "未绑定视频号",
                8: "主体未验证",    # 企业资质没审核通过
                9: "未绑定视频号且主体未验证",
                10: "用户拒收"      # 用户把我们拉黑啦！！！
            }
            
            reason = fail_reasons.get(fail_type, "未知原因")
            logger.warning(f"消息发送失败 - 用户: {external_userid}, 消息ID: {fail_msgid}, 原因: {reason}")
            
        except Exception as e:
            logger.error(f"处理发送失败事件异常: {e}")
    
    def handle_servicer_status_change_event(self, event_data: Dict) -> None:
        """处理接待人员状态变更事件，暂不处理，仅做日志记录"""
        try:
            servicer_userid = event_data.get('servicer_userid', '')  # 接待人员的企业微信ID
            status = event_data.get('status', 0)                     # 取新状态编号
            open_kfid = event_data.get('open_kfid', '')              # 取客服账号ID
            
            logger.info(f"接待人员 {servicer_userid} 状态变更: {status}, open_kfid: {open_kfid}")
            
        except Exception as e:
            logger.error(f"处理接待人员状态变更事件异常: {e}")
    
    # 处理"会话状态变更"事件（这个比较重要，要在状态变化时给用户发提示）
    def handle_session_change_event(self, event_data: Dict) -> None:
        """处理会话状态变更，回复欢迎语或者结束语"""
        try:
            external_userid = event_data.get('external_userid', '') # 取用户ID
            change_type = event_data.get('change_type', 0)          # 取变更类型编号
            msg_code = event_data.get('msg_code', '')               # 消息码（跟welcome_code类似，是微信给的临时发送凭证）
            
            change_types = {
                1: "从接待池接入会话",  # 人工客服刚接起这个会话
                2: "转接会话", 
                3: "结束会话",
                4: "重新接入已结束/已转接会话"
            }
            
            change_text = change_types.get(change_type, "未知变更")
            logger.info(f"会话状态变更 - 用户: {external_userid}, 变更: {change_text}")
            
            # 如果有消息码，可以发送相应的回复语或结束语
            if msg_code:
                if change_type == 1:  # 接入会话
                    response_msg = "您好！我是您的专属客服，很高兴为您服务！有什么可以帮助您的吗？"
                elif change_type == 3:  # 结束会话
                    response_msg = "感谢您的咨询！如有其他问题，欢迎随时联系我们。祝您生活愉快！"
                else:
                    response_msg = None
                
                if response_msg:
                    wecom_api.send_event_response_message(msg_code, response_msg)
            
        except Exception as e:
            logger.error(f"处理会话状态变更事件异常: {e}")
    
    def handle_user_recall_event(self, event_data: Dict) -> None:
        """处理用户撤回消息事件，暂不处理，仅做日志记录"""
        try:
            external_userid = event_data.get('external_userid', '') # 用户ID
            recall_msgid = event_data.get('recall_msgid', '')       # 被撤回的消息ID
            
            logger.info(f"用户 {external_userid} 撤回消息: {recall_msgid}")
            
        except Exception as e:
            logger.error(f"处理用户撤回消息事件异常: {e}")
    
    def handle_servicer_recall_event(self, event_data: Dict) -> None:
        """处理接待人员撤回消息事件，暂不处理，仅做日志记录"""
        try:
            external_userid = event_data.get('external_userid', '') # 用户ID
            servicer_userid = event_data.get('servicer_userid', '') # 接待人员ID
            recall_msgid = event_data.get('recall_msgid', '')       # 被撤回的消息ID
            
            logger.info(f"接待人员 {servicer_userid} 撤回消息: {recall_msgid}")
            
        except Exception as e:
            logger.error(f"处理接待人员撤回消息事件异常: {e}")
        
    def get_smart_reply(self, content: str) -> str:
        """智能回复，使用RAG服务获取回复内容，并记录token消耗情况和参考文档"""
        logger.info(f"智能客服回复开始")

        try:
            # 调用智能客服接口，获取智能回复内容和token消耗情况
            # ======================单例模式的全局获取RAG全局单例的方法===============
            response = get_rag_service().query_service(content)

            if response.success:
                # 记录token使用情况
                if response.token_usage:
                    logger.info(f"💰 本次查询Token使用: 输入{response.token_usage.prompt_tokens}, "
                            f"输出{response.token_usage.completion_tokens}, "
                            f"总计{response.token_usage.total_tokens}")
                
                # 记录参考文档
                if response.source_documents:
                    logger.info(f"📚 参考文档: {[doc.doc_name for doc in response.source_documents]}")
                
                return response.content
            else:
                logger.error(f"RAG查询失败: {response.error_message}")
                return "抱歉，我暂时无法理解您的问题，请稍后再试或联系人工客服。"

        except Exception as e:
            logger.error(f"智能回复处理异常: {e}")
            return "抱歉，我暂时无法理解您的问题，请稍后再试或联系人工客服。"
    
    
    def send_kf_reply(self, external_userid: str, open_kfid: str, content: str) -> bool:
        """发送客服回复"""
        try:
            # 检查回复内容长度
            if len(content) > Config.KF_MAX_REPLY_LENGTH:
                content = content[:Config.KF_MAX_REPLY_LENGTH-10] + "...(内容过长已截断)"
            
            # 调用微信API发送文字消息
            result = wecom_api.send_kf_text_message(external_userid, open_kfid, content)
            if result:
                logger.info(f"发送客服回复成功 - 用户: {external_userid}")
                return True
            else:
                logger.error(f"发送客服回复失败 - 用户: {external_userid}")
                return False
        except Exception as e:
            logger.error(f"发送客服回复异常: {e}")
            return False
    

    def get_service_session_state(self, external_userid: str, open_kfid: str) -> int:
        """获取服务会话状态"""
        try:
            # 调用微信API查询这个用户的会话状态
            result = wecom_api.get_service_session_state(external_userid, open_kfid)
            if result:
                logger.info(f"获取服务会话状态成功 - 用户: {external_userid}")
                logger.info(f"服务会话状态: {result}")
                return result.get('service_state', -1)
            else:
                logger.error(f"获取服务会话状态失败 - 用户: {external_userid}")
                return -1
        except Exception as e:
            logger.error(f"获取服务会话状态异常: {e}")
            return -1

    def update_service_session_state_to_service_pool(self, external_userid: str, open_kfid: str) -> bool:
        """更新服务会话状态"""
        try:
            result = wecom_api.update_service_session_state(external_userid, open_kfid, 2) # 2: 转接会话进入接待池，后续由人工接待
            if result:
                logger.info(f"更新服务会话状态成功 - 用户: {external_userid}")
                return True
            else:
                logger.error(f"更新服务会话状态失败 - 用户: {external_userid}")
                return False
        except Exception as e:
            logger.error(f"更新服务会话状态异常: {e}")
            return False
    

# 全局客服消息处理器实例
kf_message_handler = KfMessageHandler() 