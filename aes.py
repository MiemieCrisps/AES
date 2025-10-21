from Crypto.Util.number import long_to_bytes, bytes_to_long
import random
import numpy as np

# ---------------------- AES核心常量与轮函数实现 ----------------------
# AES S盒（字节替换表）
S_BOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

# AES轮常量（用于密钥扩展）
RC = [0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000,
      0x20000000, 0x40000000, 0x80000000, 0x1B000000, 0x36000000]


def sub_bytes(state):
    """字节替换：使用S盒替换状态中的每个字节（确保输入为uint8）"""
    # 强制转换为uint8，防止值溢出
    state = state.astype(np.uint8)
    return np.vectorize(lambda x: S_BOX[x])(state).astype(np.uint8)


def shift_rows(state):
    """行移位：每行循环左移（第1行1位，第2行2位，第3行3位）"""
    return np.array([
        [state[0, 0], state[0, 1], state[0, 2], state[0, 3]],  # 第0行不移位
        [state[1, 1], state[1, 2], state[1, 3], state[1, 0]],  # 第1行左移1位
        [state[2, 2], state[2, 3], state[2, 0], state[2, 1]],  # 第2行左移2位
        [state[3, 3], state[3, 0], state[3, 1], state[3, 2]]  # 第3行左移3位
    ], dtype=np.uint8)  # 显式指定uint8


def mix_columns(state):
    """列混合：通过矩阵乘法混合每列的字节（最后一轮不执行）"""

    def mul2(x):
        """GF(2^8)中乘以2的运算（确保结果在0~255）"""
        res = (x << 1) if (x & 0x80) == 0 else (x << 1) ^ 0x1B
        return res & 0xFF  # 强制截断为8位（0~255）

    def mix_col(col):
        """混合单个列"""
        a, b, c, d = col
        return [
            mul2(a) ^ mul2(b) ^ b ^ c ^ d,
            a ^ mul2(b) ^ mul2(c) ^ c ^ d,
            a ^ b ^ mul2(c) ^ mul2(d) ^ d,
            mul2(a) ^ a ^ b ^ c ^ mul2(d)
        ]

    # 对每列执行混合操作，确保结果为uint8
    mixed = np.array([mix_col(col) for col in state.T], dtype=np.uint8).T
    return mixed.astype(np.uint8)


def add_round_key(state, round_key):
    """轮密钥加：状态与轮密钥逐字节异或（确保结果为uint8）"""
    return (state ^ round_key).astype(np.uint8)


def key_expansion(key):
    """密钥扩展：从原始密钥生成所有轮的密钥（AES-128共11轮密钥）"""
    # 将16字节密钥转换为4个32位字
    key_words = np.array([bytes_to_long(key[i * 4:(i + 1) * 4]) for i in range(4)], dtype=np.uint32)
    w = [0] * 44  # 11轮密钥 × 4字/轮 = 44字

    # 初始化前4个字
    for i in range(4):
        w[i] = key_words[i]

    # 生成剩余的40个字
    for i in range(4, 44):
        temp = w[i - 1]
        if i % 4 == 0:
            # 轮常量异或 + 字节替换 + 循环左移
            temp = (temp << 8) | (temp >> 24)  # 循环左移1字节
            # 对每个字节应用S盒（确保每个字节在0~255）
            temp = (S_BOX[(temp >> 24) & 0xFF] << 24 |
                    S_BOX[(temp >> 16) & 0xFF] << 16 |
                    S_BOX[(temp >> 8) & 0xFF] << 8 |
                    S_BOX[temp & 0xFF])
            temp ^= RC[i // 4 - 1]  # 异或轮常量
        w[i] = w[i - 4] ^ temp

    # 将44个字转换为11个4×4的轮密钥矩阵（确保为uint8）
    round_keys = []
    for i in range(11):
        key_bytes = b''.join([long_to_bytes(w[i * 4 + j], 4) for j in range(4)])
        round_key = np.frombuffer(key_bytes, dtype=np.uint8).reshape(4, 4)
        round_keys.append(round_key)
    return round_keys


# ---------------------- 比特差异统计工具函数 ----------------------
def count_bit_differences(a, b):
    """统计两个字节串的比特差异数"""
    if len(a) != len(b):
        raise ValueError("输入字节串长度必须相同")
    # 转换为整数后异或，统计1的个数
    diff = int.from_bytes(a, 'big') ^ int.from_bytes(b, 'big')
    return bin(diff).count('1')


# ---------------------- 明文雪崩效应测试（固定密钥，明文1-bit差异） ----------------------
def test_plaintext_avalanche():
    key_length = 128  # 仅测试AES-128
    key = bytes([0] * (key_length // 8))  # 固定全0密钥
    block_size = 16  # AES分组长度16字节
    plaintext = bytes([0] * block_size)  # 初始明文（全0）
    round_counts = 10  # AES-128共10轮
    total_diffs = [0] * (round_counts + 1)  # 索引0~10（含初始状态）
    num_pairs = 100  # 测试100对明文（提高统计准确性）

    # 预生成轮密钥（仅需生成一次）
    round_keys = key_expansion(key)

    for _ in range(num_pairs):
        # 生成仅1-bit不同的明文对
        bit_pos = random.randint(0, block_size * 8 - 1)  # 随机比特位置
        byte_idx = bit_pos // 8
        bit_idx = bit_pos % 8

        # 构造原始明文m和差异明文m'
        m = bytearray(plaintext)
        m_prime = bytearray(plaintext)
        m_prime[byte_idx] ^= (1 << bit_idx)  # 翻转指定比特
        m, m_prime = bytes(m), bytes(m_prime)

        # 初始状态（第0轮，未加密）：显式指定uint8
        state_m = np.frombuffer(m, dtype=np.uint8).reshape(4, 4).astype(np.uint8)
        state_m_prime = np.frombuffer(m_prime, dtype=np.uint8).reshape(4, 4).astype(np.uint8)
        diff = count_bit_differences(
            state_m.tobytes(),
            state_m_prime.tobytes()
        )
        total_diffs[0] += diff

        # 执行第1轮到第10轮加密，并统计每轮差异
        for round_idx in range(1, round_counts + 1):
            # 1. 字节替换（SubBytes）
            state_m = sub_bytes(state_m)
            state_m_prime = sub_bytes(state_m_prime)

            # 2. 行移位（ShiftRows）
            state_m = shift_rows(state_m)
            state_m_prime = shift_rows(state_m_prime)

            # 3. 列混合（MixColumns）：最后一轮不执行
            if round_idx != round_counts:
                state_m = mix_columns(state_m)
                state_m_prime = mix_columns(state_m_prime)

            # 4. 轮密钥加（AddRoundKey）
            state_m = add_round_key(state_m, round_keys[round_idx])
            state_m_prime = add_round_key(state_m_prime, round_keys[round_idx])

            # 统计当前轮的比特差异
            diff = count_bit_differences(
                state_m.tobytes(),
                state_m_prime.tobytes()
            )
            total_diffs[round_idx] += diff

    # 计算每轮的平均差异
    avg_diffs = [d / num_pairs for d in total_diffs]
    print("明文雪崩效应测试（固定密钥，明文1-bit差异）")
    for i, diff in enumerate(avg_diffs):
        print(f"第{i}轮，平均比特差异：{diff:.2f}")


# ---------------------- 密钥雪崩效应测试（固定明文，密钥1-bit差异） ----------------------
def test_key_avalanche():
    key_length = 128  # 仅测试AES-128
    plaintext = bytes([0] * 16)  # 固定全0明文
    round_counts = 10  # AES-128共10轮
    total_diffs = [0] * (round_counts + 1)  # 索引0~10
    num_pairs = 100  # 测试100对密钥

    for _ in range(num_pairs):
        # 生成仅1-bit不同的密钥对
        bit_pos = random.randint(0, key_length - 1)  # 随机密钥比特位置
        byte_idx = bit_pos // 8
        bit_idx = bit_pos % 8

        # 构造原始密钥k和差异密钥k'
        key = bytes([0] * (key_length // 8))
        key_prime = bytearray(key)
        key_prime[byte_idx] ^= (1 << bit_idx)  # 翻转指定比特
        key, key_prime = bytes(key), bytes(key_prime)

        # 生成两组轮密钥（原始密钥和差异密钥）
        round_keys = key_expansion(key)
        round_keys_prime = key_expansion(key_prime)

        # 初始状态（第0轮，明文相同，差异为0）：显式指定uint8
        state = np.frombuffer(plaintext, dtype=np.uint8).reshape(4, 4).astype(np.uint8)
        state_prime = np.frombuffer(plaintext, dtype=np.uint8).reshape(4, 4).astype(np.uint8)
        diff = count_bit_differences(
            state.tobytes(),
            state_prime.tobytes()
        )
        total_diffs[0] += diff

        # 执行第1轮到第10轮加密，使用各自的轮密钥
        for round_idx in range(1, round_counts + 1):
            # 原始密钥加密路径
            state = sub_bytes(state)
            state = shift_rows(state)
            if round_idx != round_counts:
                state = mix_columns(state)
            state = add_round_key(state, round_keys[round_idx])

            # 差异密钥加密路径
            state_prime = sub_bytes(state_prime)
            state_prime = shift_rows(state_prime)
            if round_idx != round_counts:
                state_prime = mix_columns(state_prime)
            state_prime = add_round_key(state_prime, round_keys_prime[round_idx])

            # 统计当前轮的比特差异
            diff = count_bit_differences(
                state.tobytes(),
                state_prime.tobytes()
            )
            total_diffs[round_idx] += diff

    # 计算每轮的平均差异
    avg_diffs = [d / num_pairs for d in total_diffs]
    print("\n密钥雪崩效应测试（固定明文，密钥1-bit差异）")
    for i, diff in enumerate(avg_diffs):
        print(f"第{i}轮，平均比特差异：{diff:.2f}")


# ---------------------- 执行测试 ----------------------
if __name__ == "__main__":
    test_plaintext_avalanche()
    test_key_avalanche()