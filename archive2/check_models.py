import urllib.request, json

req = urllib.request.Request("https://openrouter.ai/api/v1/models")
with urllib.request.urlopen(req, timeout=10) as resp:
    data = json.loads(resp.read())
    models = data.get("data", [])
    for m in models[:30]:
        print(f"{m['id']} - {m.get('name', '')}")
