✅ Example: Serve Streamlit using hostname
If your machine's hostname is, for example, myserver.domain.com, you can run:

streamlit run your_app.py --server.address myserver.domain.com --server.port 8501
Make sure:

myserver.domain.com resolves to your machine's IP (e.g. via DNS or /etc/hosts).

Your firewall or cluster allows access to port 8501.

You are not behind NAT/firewall that blocks incoming connections.

💡 Alternative (make available on all network interfaces)
To allow access via hostname or IP from other devices on the same network:

bash
Copy
Edit
streamlit run your_app.py --server.address 0.0.0.0 --server.port 8501
Then access it via:

IP address: http://your_ip:8501

Hostname: http://your_hostname:8501

🛠️ Optional: Set permanent config
You can avoid passing --server.address every time by setting it in Streamlit config:

bash
Copy
Edit
streamlit config show > ~/.streamlit/config.toml
Edit the file ~/.streamlit/config.toml:

toml
Copy
Edit
[server]
address = "0.0.0.0"
port = 8501
enableCORS = false
