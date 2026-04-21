#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
微信客服消息cursor持久化管理模块
用于管理每个客服账号的消息拉取cursor，避免重复处理消息
"""

import os
import json
import logging
import sqlite3    # 导入Python内置的SQLite数据库驱动，不需要装第三方库就能直接操作本地数据库文件
import threading  # 导入多线程模块，用来创建“锁”，防止多个线程同时改数据库导致数据错乱
from typing import Dict, Optional
from config import Config

logger = logging.getLogger(__name__)

# CursorManager（游标管理器）类
class CursorManager:
    """Cursor持久化管理器"""
    
    def __init__(self, db_path: str = None):   # 传入自定义的数据库路径，默认为空
        self.db_path = db_path or Config.DATABASE_PATH # .env: DATABASE_PATH=messages.db
        self.lock = threading.Lock()  # 创建一把线程锁，存到对象属性里（相当于给数据库柜子加了一把锁）
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:     # 连接数据库文件，如果文件不存在会自动创建，with语句结束时自动关闭连接
                cursor = conn.cursor()  # 创建一个游标对象（这里的游标是数据库概念，相当于拿笔签字的人，不是企业微信的那个cursor）
                
                # 创建cursor存储表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS kf_cursors (
                        open_kfid TEXT PRIMARY KEY,
                        cursor TEXT NOT NULL,
                        last_updated INTEGER NOT NULL,
                        message_count INTEGER DEFAULT 0,
                        created_time INTEGER DEFAULT (strftime('%s', 'now'))
                    )
                ''')
                
                # 创建消息去重表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processed_messages (
                        msgid TEXT PRIMARY KEY,
                        open_kfid TEXT NOT NULL,
                        external_userid TEXT,
                        msgtype TEXT,
                        origin INTEGER,
                        processed_time INTEGER DEFAULT (strftime('%s', 'now')),
                        reply_sent INTEGER DEFAULT 0
                    )
                ''') # reply_sent：未回复 / 已回复
                
                # 创建索引
                # 把 processed_time 这一列的数据抽出来，排好序，构建了一棵B-树
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_processed_messages_time 
                    ON processed_messages(processed_time)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_processed_messages_kfid 
                    ON processed_messages(open_kfid)
                ''')
                
                conn.commit() # 把上面的建表、建索引操作正式提交到数据库文件中
                logger.info("Cursor管理数据库初始化完成")
                
        except Exception as e:
            logger.error(f"初始化cursor数据库失败: {e}")
    
    # 定义获取“拉取进度”的方法，传入客服ID，返回字符串
    def get_cursor(self, open_kfid: str) -> str:
        """
        获取指定客服账号的cursor
        
        Args:
            open_kfid: 客服账号ID
            
        Returns:
            cursor字符串，如果不存在则返回空字符串
        """
        try:
            with self.lock: # 🔒关键：先拿到锁。如果别的线程正在用数据库，这里会排队等待，拿到锁了才往下走
                with sqlite3.connect(self.db_path) as conn: # 连接数据库
                    cursor = conn.cursor() # 
                    # 执行查询
                    cursor.execute(
                        'SELECT cursor FROM kf_cursors WHERE open_kfid = ?', # 从kf_cursors表里查cursor列，条件是客服ID等于传入的值
                        (open_kfid,)  # 元组，里面装的是替换上面问号?的真实数据，防止SQL注入攻击
                    )
                    result = cursor.fetchone() # 只取查询结果的第一条数据（因为主键唯一，最多只有一条）
                    
                    if result:
                        logger.debug(f"获取cursor成功 - {open_kfid}: {result[0][:20]}...")
                        return result[0] # 返回查到的cursor字符串（result是一个元组，result[0]就是第一列的值）
                    else: # 如果没查到数据（说明这个客服账号第一次用）
                        logger.info(f"客服账号 {open_kfid} 无历史cursor，将进行首次拉取")
                        return ""
                        
        except Exception as e:
            logger.error(f"获取cursor失败: {e}")
            return ""
    
    # 定义保存“拉取进度”的方法，传入客服ID、新cursor、消息数量，默认0，返回布尔值
    def save_cursor(self, open_kfid: str, cursor: str, message_count: int = 0) -> bool:
        """
        保存指定客服账号的cursor
        
        Args:
            open_kfid: 客服账号ID
            cursor: 新的cursor值
            message_count: 本次处理的消息数量
            
        Returns:
            保存是否成功
        """
        try:
            with self.lock: # 🔒拿到线程锁
                with sqlite3.connect(self.db_path) as conn: # 连接数据库
                    db_cursor = conn.cursor() # 为了不和参数名cursor冲突，把数据库游标改名为db_cursor
                    
                    # 使用UPSERT语法，“有则更新，无则插入”，然后子查询：趁当前这行还没被删掉前，赶紧把它现在的 `message_count` 查出来
                    db_cursor.execute('''
                        INSERT OR REPLACE INTO kf_cursors 
                        (open_kfid, cursor, last_updated, message_count)
                        VALUES (?, ?, strftime('%s', 'now'), 
                               COALESCE((SELECT message_count FROM kf_cursors WHERE open_kfid = ?), 0) + ?)
                    ''', (open_kfid, cursor, open_kfid, message_count))  # 按顺序替换SQL里的四个问号
                    
                    conn.commit() # 把上面的UPSERT操作正式提交到数据库文件中
                    logger.info(f"保存cursor成功 - {open_kfid}: {cursor[:20]}..., 消息数: {message_count}")
                    return True
                    
        except Exception as e:
            logger.error(f"保存cursor失败: {e}")
            return False
    
    def is_message_processed(self, msgid: str) -> bool:
        """
        检查消息是否已经处理过
        
        Args:
            msgid: 消息ID
            
        Returns:
            是否已处理
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor() # 创建游标
                    cursor.execute(  # 执行查询
                        'SELECT 1 FROM processed_messages WHERE msgid = ?', # 查msgid列，SELECT 1的意思是“只要查到了就行，不在乎具体内容”，性能最好
                        (msgid,) # 替换问号
                    )
                    result = cursor.fetchone() # 取结果
                    return result is not None
                    
        except Exception as e:
            logger.error(f"检查消息处理状态失败: {e}")
            return False
    
    def mark_message_processed(self, msgid: str, open_kfid: str, external_userid: str = "", 
                             msgtype: str = "", origin: int = 0, reply_sent: bool = False) -> bool:
        """
        标记消息为已处理
        
        Args:
            msgid: 消息ID
            open_kfid: 客服账号ID
            external_userid: 外部用户ID
            msgtype: 消息类型
            origin: 消息来源
            reply_sent: 是否已发送回复
            
        Returns:
            标记是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    # 遇到重复msgid就覆盖、指定列、问号占位
                    cursor.execute('''
                        INSERT OR REPLACE INTO processed_messages 
                        (msgid, open_kfid, external_userid, msgtype, origin, reply_sent)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (msgid, open_kfid, external_userid, msgtype, origin, int(reply_sent)))  # 把Python的布尔值reply_sent转成整数(0或1)塞进去
                    
                    conn.commit()
                    logger.debug(f"标记消息已处理: {msgid}")
                    return True
                    
        except Exception as e:
            logger.error(f"标记消息处理状态失败: {e}")
            return False
    
    def cleanup_old_records(self, days: int = 30) -> bool:
        """
        清理旧的处理记录
        
        Args:
            days: 保留天数
            
        Returns:
            清理是否成功
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 清理旧的消息处理记录，条件是：处理时间小于“当前时间减去N天”的时间戳
                    cursor.execute('''
                        DELETE FROM processed_messages 
                        WHERE processed_time < strftime('%s', 'now', '-{} days')
                    '''.format(days)) # 用format把变量days填进SQL字符串的{}里
                    
                    deleted_count = cursor.rowcount # 获取刚才删除了几行数据
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"清理了 {deleted_count} 条旧的消息处理记录")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        获取cursor管理统计信息
        
        Returns:
            统计信息字典
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取客服账号数量
                    cursor.execute('SELECT COUNT(*) FROM kf_cursors')
                    kf_count = cursor.fetchone()[0]  # 取出总数
                    
                    # 获取总消息处理数量
                    cursor.execute('SELECT SUM(message_count) FROM kf_cursors')
                    total_messages = cursor.fetchone()[0] or 0  # 如果数据库是空的没数据，fetchone()[0]会是None，用or 0保证它是整数0
                    
                    # 获取今日处理消息数量，条件：处理时间大于等于今天0点的时间戳
                    cursor.execute('''
                        SELECT COUNT(*) FROM processed_messages 
                        WHERE processed_time >= strftime('%s', 'now', 'start of day')
                    ''')
                    today_messages = cursor.fetchone()[0]
                    
                    # 获取已回复消息数量，条件：reply_sent为1，即已发送回复
                    cursor.execute('SELECT COUNT(*) FROM processed_messages WHERE reply_sent = 1')
                    replied_messages = cursor.fetchone()[0]
                    
                    return {
                        'kf_accounts': kf_count,                # 客服数量
                        'total_messages': total_messages,       # 总消息处理数量
                        'today_messages': today_messages,       # 今日处理消息数量
                        'replied_messages': replied_messages,   # 已回复消息数量
                        # 键：回复率。分母用max(total, 1)防止除以0报错，乘100转百分比，round保留两位小数
                        'reply_rate': round(replied_messages / max(total_messages, 1) * 100, 2)
                    }
                    
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}
    
    def get_kf_account_info(self, open_kfid: str) -> Optional[Dict]:
        """
        获取指定客服账号的详细信息
        
        Args:
            open_kfid: 客服账号ID
            
        Returns:
            账号信息字典
        """
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # 获取cursor信息，从进度表查，条件：open_kfid指定客服账号ID
                    cursor.execute('''
                        SELECT cursor, last_updated, message_count, created_time 
                        FROM kf_cursors WHERE open_kfid = ?
                    ''', (open_kfid,))
                    cursor_info = cursor.fetchone() # 取出这一行数据
                    
                    if not cursor_info:
                        return None
                    
                    # 获取今日消息数量，条件：open_kfid指定客服账号ID，处理时间大于等于今天0点的时间戳
                    cursor.execute('''
                        SELECT COUNT(*) FROM processed_messages 
                        WHERE open_kfid = ? AND processed_time >= strftime('%s', 'now', 'start of day')
                    ''', (open_kfid,))
                    today_count = cursor.fetchone()[0]
                    
                    return {
                        'open_kfid': open_kfid,                 # 客服账号ID
                        'cursor': cursor_info[0][:20] + '...' if cursor_info[0] else '', # cursor值，最多20个字符
                        'last_updated': cursor_info[1],         # 最后更新时间
                        'total_messages': cursor_info[2],       # 总消息处理数量
                        'today_messages': today_count,          # 今日处理消息数量
                        'created_time': cursor_info[3]          # 创建时间
                    }
                    
        except Exception as e:
            logger.error(f"获取客服账号信息失败: {e}")
            return None

# 全局cursor管理器实例
cursor_manager = CursorManager() 