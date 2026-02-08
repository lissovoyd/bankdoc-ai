import redis

r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
r.set("test", "hello")
print(r.get("test"))
print("✓ Redis connection works!")