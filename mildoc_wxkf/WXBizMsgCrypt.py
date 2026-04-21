#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################
# Author: me
# Created Time: 
# File Name: WXBizMsgCrypt.py
# Description: 企业微信回调消息加解密
#########################################################################

import base64                       # 二进制乱码 -> 字符串
import string
import random
import hashlib                      # 导入哈希算法模块，里面有SHA1算法，用来算签名指纹
import time
import struct                       # 导入结构体模块，这是一个极其底层的库，专门用来把数字按照C语言的方式打包成严格的字节流
from Crypto.Cipher import AES       # 导入AES加密算法库
import xml.etree.cElementTree as ET # 导入XML解析器，微信传数据喜欢用XML格式包裹

"""
关于Crypto.Cipher模块，ImportError: No module named 'Crypto'解决方案
请到官方网站 https://www.dlitz.net/software/pycrypto/ 下载pycrypto。
下载后，按照README中的"Installation"小节的提示进行pycrypto安装。
"""
class FormatException(Exception): # 自定义一个异常类，专门用来表示格式不对这种错误
    pass # 什么都不用写，继承自带的Exception就行

def throw_exception(message, exception_class=FormatException):
    """抛出异常信息"""
    raise exception_class(message)  # 强行让程序报错退出，并打印错误信息

class SHA1:
    """计算企业微信签名的SHA1算法"""

    def getSHA1(self, token, timestamp, nonce, encrypt):
        """用SHA1算法生成安全签名
        @param token:  票据
        @param timestamp: 时间戳
        @param nonce: 随机字符串
        @param encrypt: 密文
        @return: 安全签名
        """
        try:
            sortlist = [token, timestamp, nonce, encrypt]
            sortlist.sort()                                 # 极其关键：按照字符串的字典序（ASCII码大小）给它们排序！比如1排在a前面，a排在b前面
            sha = hashlib.sha1()                            # 创建一个SHA1哈希加工器
            sha.update("".join(sortlist).encode('utf-8'))   # 把排好序的四个东西拼成一长串字符串，转成utf-8字节，扔进加工器
            return sha.hexdigest()                          # 启动加工器，吐出一串40个字符的十六进制乱码（这就是签名指纹）
        except Exception as e:
            print(e)
            return ""

class XMLParse: # XML处理类，微信发来的数据外面包着一层XML皮
    """提供接口：提取消息格式中的密文及生成回复消息格式"""

    # xml消息模板（下面这段是用来回复微信时，包在外面的XML壳子）
    AES_TEXT_RESPONSE_TEMPLATE = """<xml>
<Encrypt><![CDATA[%(msg_encrypt)s]]></Encrypt>
<MsgSignature><![CDATA[%(msg_signaturet)s]]></MsgSignature>
<TimeStamp>%(timestamp)s</TimeStamp>
<Nonce><![CDATA[%(nonce)s]]></Nonce>
</xml>"""

    # 从微信发来的XML里拆出里面的密文的方法
    def extract(self, xmltext):
        """提取出xml数据包中的加密消息
        @param xmltext: 待提取的xml字符串
        @return: 提取出的加密消息字符串
        """
        try:
            xml_tree = ET.fromstring(xmltext)           # 把XML字符串（POST请求的数据）解析成一棵树
            encrypt = xml_tree.find("Encrypt")
            touser_name = xml_tree.find("ToUserName")   # 接收者
            return 0, encrypt.text, touser_name.text
        except Exception as e:
            print(e)
            return -40003, None, None

    # 把密文重新打包成XML格式的方法
    def generate(self, encrypt, signature, timestamp, nonce):
        """生成xml消息
        @param encrypt: 加密后的消息
        @param signature: 安全签名
        @param timestamp: 时间戳
        @param nonce: 随机字符串
        @return: 生成的xml字符串
        """
        resp_dict = {
            'msg_encrypt': encrypt,
            'msg_signaturet': signature,
            'timestamp': timestamp,
            'nonce': nonce,
        }
        # 字典数据填入模板（老式的字符串格式化语法）
        resp_xml = self.AES_TEXT_RESPONSE_TEMPLATE % resp_dict
        return resp_xml

class PKCS7Encoder():
    """提供基于PKCS7算法的加解密接口"""

    # 定义数据块的长度必须是32个字节（注意：标准AES是16字节，但微信这里硬性规定按32字节补齐）
    block_size = 32

    def encode(self, text):
        """ 对需要加密的明文进行填充补位
        @param text: 需要进行填充补位操作的明文
        @return: 补齐明文字符串
        """
        text_length = len(text)
        # 计算需要填充的位数，用32减去余数。比如长10字节，32-10=22，需要补22个字节
        amount_to_pad = self.block_size - (text_length % self.block_size)
        if amount_to_pad == 0:  # 如果数据长度刚好是32的整数倍（余数为0）
            amount_to_pad = self.block_size # 也必须补32个字节！（这是PKCS7的硬性规定，为了解密时好识别）
        
        # 处理bytes和str类型的兼容性
        if isinstance(text, bytes):
            # 生成一堆值为amount_to_pad的字节（比如要补22个，就生成22个值为22的字节）
            # 补的字节内容就是“amount_to_pad的值”，即amount_to_pad个”amount_to_pad“，解密方便
            pad = bytes([amount_to_pad] * amount_to_pad)
            return text + pad   # 放入原数据之后
        else:   # 若是str
            # 获得补位所用的字符
            pad = chr(amount_to_pad)  # 把数字转成对应的ASCII字符（比如22对应ASCII里的控制字符）
            return text + pad * amount_to_pad

    def decode(self, decrypted):
        """删除解密后明文的补位字符
        @param decrypted: 解密后的明文
        @return: 删除补位字符后的明文
        """
        # 兼容Python 3.6：处理bytes和str类型
        if isinstance(decrypted[-1], int): # Python 3里，取字节串最后一个元素，它是个整数
            pad = decrypted[-1]  # 在Python 3中，bytes的索引返回int。直接拿这个整数，这个整数代表补了多少个字节
        else:
            pad = ord(decrypted[-1])  # 在Python 2中或str类型时需要ord()，把字符转为对应的ASCII数字
            
        if pad < 1 or pad > 32: # 如果算出来的补位数不在1到32之间，说明数据坏了
            pad = 0 # 当作没补位处理，防止切错把真数据切没了
        return decrypted[:-pad] # 切片：从开头切到倒数第pad个，直接把尾巴上的补丁全扔掉

class Prpcrypt(object):
    """核心加解密引擎类：提供接收和推送给企业微信消息的加解密接口"""

    def __init__(self, key):
        # self.key = base64.b64decode(key+"=")
        self.key = key # 把微信后台配置的AES密钥（已经在外部转成字节了）存起来
        # 设置加解密模式为AES的CBC模式
        self.mode = AES.MODE_CBC

    def encrypt(self, text, receiveid):
        """对明文进行加密
        @param text: 需要加密的明文
        @return: 加密得到的字符串
        """
        # 16位随机字符串添加到明文开头
        text = text.encode('utf-8') # 先把字符串转为字节
        # 使用大端序（网络字节序）打包长度字段，与企业微信标准保持一致。前四个字节是明文的长度，后面才是正文
        text = self.get_random_str() + struct.pack(">I", len(text)) + text + receiveid.encode('utf-8') # 拼装：随机数 + 长度 + 正文 + 接收者ID
        # 使用自定义的填充方式对明文进行补位填充
        pkcs7 = PKCS7Encoder()
        text = pkcs7.encode(text) # 对明文进行填充补位（32位）
        # 创建AES加密器：传入密钥、CBC模式、初始向量（微信规定直接取密钥的前16个字节当初始向量）
        cryptor = AES.new(self.key, self.mode, self.key[:16])
        try:
            ciphertext = cryptor.encrypt(text) # 启动加密器，吐出密文
            # 使用BASE64对加密后的字符串进行编码，方便在网络上传输（因为纯字节容易丢失）
            return base64.b64encode(ciphertext)
        except Exception as e:
            print(e)
            return None

    def decrypt(self, text, receiveid, verify_receiveid=True):
        """对解密后的明文进行补位删除
        @param text: 密文
        @param receiveid: 接收者ID
        @param verify_receiveid: 是否验证receiveid，URL验证时设为False
        @return: 删除填充补位后的明文
        """
        try:
            # 与加密器相同参数，创建解密器
            cryptor = AES.new(self.key, self.mode, self.key[:16])
            # 使用BASE64对密文进行解码，然后AES-CBC解密
            plain_text = cryptor.decrypt(base64.b64decode(text)) # 先把Base64转回字节，再丢进解密器还原出带补丁的明文
        except Exception as e:
            print(e)
            return None
        try:
            # 去除补位字符
            pkcs7 = PKCS7Encoder()
            plain_text = pkcs7.decode(plain_text) # 把尾巴上的补丁切掉
            # 去除16位随机字符串，即get_random_str()
            content = plain_text[16:] # 1. 把开头我们自己加的16个随机字符切掉
            # 2. 读取正文长度。使用大端序（网络字节序）解析长度字段，这是企业微信使用的标准
            xml_len = struct.unpack(">I", content[:4])[0] # 把开头4个字节转回整数（这就是正文的长度），[0]是因为unpack返回的是个元组
            xml_content = content[4:xml_len + 4] # 3. 把正文切出来
            from_receiveid = content[xml_len + 4:] # 4. 把剩下的receiveid接收者ID切出来
        except Exception as e:
            print(e)
            return None
        
        # 如果需要验证receiveid，则进行验证。收件人地址，企业ID（Corpid）
        if verify_receiveid and from_receiveid.decode('utf-8') != receiveid: # 不是URL验证截断，检查receiveid（拆出来的ID）是否匹配我们配置好的ID
            return None # 不一样说明这消息不是发给这里的，直接丢弃防伪造。收件人不是我！丢弃！
        
        return xml_content

    def get_random_str(self):
        """ 随机生成16位字符串
        @return: 16位随机字符串
        """
        # 从所有大小写字母和数字里随机抽16个拼起来并转成字节
        return "".join(random.choice(string.ascii_letters + string.digits) for _ in range(16)).encode('utf-8')

# 对外提供服务的总门面类
class WXBizMsgCrypt(object):
    # 构造函数
    # @param sToken: 企业微信后台，开发者设置的Token
    # @param sEncodingAESKey: 企业微信后台，开发者设置的EncodingAESKey
    # @param sReceiveId: 企业微信的CorpId 或者 应用的AgentId
    def __init__(self, sToken, sEncodingAESKey, sReceiveId):
        try:
            # 微信后台给的AESKey少了一个等号，补上等号后用Base64解码成真正的32字节密钥
            self.key = base64.b64decode(sEncodingAESKey + "=")
            assert len(self.key) == 32 # 不是32字节就报错
        except:
            throw_exception("[error]: EncodingAESKey unvalid !", FormatException)
            # return WXBizMsgCrypt_ERROR_InvalidAesKey)
        self.token = sToken
        self.receiveid = sReceiveId

        # 验证URL
        # @param sMsgSignature: 签名串，对应URL参数的msg_signature
        # @param sTimeStamp: 时间戳，对应URL参数的timestamp
        # @param sNonce: 随机串，对应URL参数的nonce
        # @param sEchoStr: 随机串，对应URL参数的echostr
        # @param sReplyEchoStr: 解密之后的echostr，当return返回0时有效
        # @return：成功0，失败返回对应的错误码

    # 验证URL参数是否合法
    def VerifyURL(self, sMsgSignature, sTimeStamp, sNonce, sEchoStr):
        sha1 = SHA1()   # 签名计算器
        ret = sha1.getSHA1(self.token, sTimeStamp, sNonce, sEchoStr) # 用”暗号、时间、随机数、加密的验证码“计算签名
        if ret != sMsgSignature: # 结果和微信传过来的指纹不一致
            return -40001
        pc = Prpcrypt(self.key) # 核心加密引擎
        # URL验证时不进行receiveid验证，只解密获取实际内容
        ret = pc.decrypt(sEchoStr, self.receiveid, verify_receiveid=False) # 解密验证码
        if ret is None: # 解密失败咯
            return -40002
        sReplyEchoStr = ret.decode('utf-8') # 把解密出的字节转成字符串
        return 0, sReplyEchoStr # 返回成功码0，和解密后的明文验证码

    # 接收日常消息专用解密方法
    def DecryptMsg(self, sPostData, sMsgSignature, sTimeStamp, sNonce):
        # 检验消息的真实性，并且获取解密后的明文
        # @param sMsgSignature: 签名串，对应URL参数的msg_signature
        # @param sTimeStamp: 时间戳，对应URL参数的timestamp
        # @param sNonce: 随机串，对应URL参数的nonce
        # @param sPostData: 密文，对应POST请求的数据
        # @param sMsg: 解密后的原文，当return返回0时有效
        # @return: 成功0，失败返回对应的错误码
        xmlParse = XMLParse()
        ret, sEncryptMsg, sToUserName = xmlParse.extract(sPostData) # 从xml里拆除密文字符串
        if ret != 0: # 拆错了
            return ret, None
        sha1 = SHA1() # 签名计算器
        ret = sha1.getSHA1(self.token, sTimeStamp, sNonce, sEncryptMsg) # 用”暗号、时间、随机数、加密的密文“重新计算签名
        if ret != sMsgSignature: 
            return -40001, None
        pc = Prpcrypt(self.key) # 核心解密引擎
        ret = pc.decrypt(sEncryptMsg, self.receiveid) # 真正解密密文，这次开启了ID验证（防伪造）
        if ret is None:
            return -40002, None
        sMsg = ret.decode('utf-8') # 转字符串
        return 0, sMsg

    # 咱们主动回复消息时的加密方法
    def EncryptMsg(self, sReplyMsg, sNonce, timestamp=None):
        # 将企业回复用户的消息加密打包
        # @param sReplyMsg: 企业号待回复用户的消息，xml格式的字符串
        # @param sTimeStamp: 时间戳，可以自己生成，也可以用URL参数的timestamp,如为None则自动用当前时间
        # @param sNonce: 随机串，可以自己生成，也可以用URL参数的nonce
        # @param sEncryptMsg: 加密后的可以直接回复用户的密文，包括msg_signature, timestamp, nonce, encrypt的xml格式的字符串,当return返回0时有效
        # return：成功0，失败返回对应的错误码
        pc = Prpcrypt(self.key) # 加密引擎
        ret = pc.encrypt(sReplyMsg, self.receiveid) # 加密明文
        if ret is None:
            return -40006, None
        if timestamp is None:
            # 外面没有传入时间戳就用当前的秒级时间
            timestamp = str(int(time.time()))
        
        # 确保ret是bytes类型，然后解码为字符串用于签名和XML生成
        if isinstance(ret, bytes):
            encrypted_str = ret.decode('utf-8') # 转字符串
        else:
            encrypted_str = str(ret)
        
        # 生成安全签名
        sha1 = SHA1()
        # 用暗号、时间、随机数、刚加密出来的密文算签名
        signature = sha1.getSHA1(self.token, timestamp, sNonce, encrypted_str)
        xmlParse = XMLParse() # XML打包器
        result_xml = xmlParse.generate(encrypted_str, signature, timestamp, sNonce)
        return 0, result_xml 