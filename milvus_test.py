from pymilvus import connections

try:
    connections.connect(host="127.0.0.1", port="19530")
    print("Milvus connection successful!")
    connections.disconnect("default")
except Exception as e:
    print(f"Milvus connection failed: {e}")