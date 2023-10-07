import socket
import json


def main():
    host = '0.0.0.0'  # 服务器的IP地址
    port = 14  # 服务器的端口号

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"等待客户端连接在 {host}:{port}...")

    client_socket, client_address = server_socket.accept()
    print(f"连接来自 {client_address}")

    messages = []  # 用于存储接收到的消息
    message_count = 0  # 计数器，用于跟踪已接收的消息数量
    file_index = 2  # 用于命名文件的索引

    while True:
        data = client_socket.recv(1024).decode("utf-8", errors="ignore")
        if not data:
            break

        try:
            message = {"tags": "", "content": data, "solution": "", "test_case": ""}
            messages.append(message)
            message_count += 1

            if message_count >= 1000:
                # 当达到500条消息时，保存到新文件，并重置计数器和消息列表
                save_messages(messages, file_index)
                messages = []
                message_count = 0
                file_index += 1

        except json.JSONDecodeError:
            print("无法解析JSON数据")

    # 保存最后不足三条的消息
    if messages:
        save_messages(messages, file_index)

    client_socket.close()
    server_socket.close()


def save_messages(messages, file_index):
    filename = f"result/ap_file_{file_index}.json"
    with open(filename, "w", encoding="utf-8") as json_file:
        json.dump(messages, json_file, indent=2, ensure_ascii=False)
    print(f"已保存到文件: {filename}")


if __name__ == "__main__":
    main()
