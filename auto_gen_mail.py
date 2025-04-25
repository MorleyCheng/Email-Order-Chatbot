import pandas as pd
import json
import random
from datetime import datetime, timedelta
import uuid

# Define products for Alex and Ben
alex_products = [
    {"name": "Wireless Keyboard", "quantity": 50, "remarks": "Black, with batteries"},
    {"name": "Wireless Mouse", "quantity": 30, "remarks": "Black, with batteries"},
    {"name": "USB-C Hub", "quantity": 20, "remarks": "4 ports"},
    {"name": "Monitor Stand", "quantity": 15, "remarks": "Adjustable height"},
    {"name": "Webcam", "quantity": 25, "remarks": "1080p"},
    {"name": "Laptop Cooler", "quantity": 10, "remarks": "RGB lighting"},
    {"name": "External SSD", "quantity": 40, "remarks": "1TB"},
    {"name": "HDMI Cable", "quantity": 60, "remarks": "2m"},
    {"name": "Desk Lamp", "quantity": 12, "remarks": "LED, dimmable"},
    {"name": "Ergonomic Chair", "quantity": 8, "remarks": "Mesh back"}
]
ben_products = [
    {"name": "LED Monitor", "quantity": 20, "remarks": "27-inch, 4K"},
    {"name": "Graphics Card", "quantity": 10, "remarks": "8GB VRAM"},
    {"name": "Mechanical Keyboard", "quantity": 15, "remarks": "RGB, mechanical"},
    {"name": "Gaming Mouse", "quantity": 25, "remarks": "Programmable buttons"},
    {"name": "Soundbar", "quantity": 12, "remarks": "Bluetooth"},
    {"name": "Network Router", "quantity": 18, "remarks": "Wi-Fi 6"},
    {"name": "Smart Speaker", "quantity": 30, "remarks": "Voice assistant"},
    {"name": "Power Strip", "quantity": 50, "remarks": "6 outlets"},
    {"name": "USB Charger", "quantity": 40, "remarks": "Fast charging"},
    {"name": "Cooling Fan", "quantity": 22, "remarks": "120mm"}
]

# Pricing for products (for vendor replies)
pricing = {
    "Wireless Keyboard": 40, "Wireless Mouse": 15, "USB-C Hub": 25, "Monitor Stand": 30,
    "Webcam": 50, "Laptop Cooler": 20, "External SSD": 100, "HDMI Cable": 10,
    "Desk Lamp": 35, "Ergonomic Chair": 150, "LED Monitor": 300, "Graphics Card": 400,
    "Mechanical Keyboard": 80, "Gaming Mouse": 45, "Soundbar": 60, "Network Router": 90,
    "Smart Speaker": 50, "Power Strip": 15, "USB Charger": 20, "Cooling Fan": 12
}

# Generate random dates between April 22, 2025, and April 30, 2025
start_date = datetime(2025, 4, 22)
end_date = datetime(2025, 4, 30)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

def random_timestamp(date):
    hour = random.randint(8, 18)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return datetime(date.year, date.month, date.day, hour, minute, second).strftime('%Y-%m-%d %H:%M:%S')

def format_markdown_table(columns, rows):
    # 計算每個欄位的最大寬度（包含中文字元）
    def get_display_width(s):
        # 計算字串顯示寬度（中文字元計為2個寬度）
        width = 0
        for char in str(s):
            width += 2 if ord(char) > 127 else 1
        return width

    # 計算每個欄位需要的寬度
    widths = []
    for i in range(len(columns)):
        col_items = [columns[i]] + [str(row[i]) for row in rows]
        widths.append(max(get_display_width(item) for item in col_items))

    # 建立表頭
    def pad_string(s, width):
        current_width = get_display_width(s)
        padding = width - current_width
        return s + " " * padding

    # 建立表頭
    header = "| " + " | ".join(pad_string(str(col), width) for col, width in zip(columns, widths)) + " |"
    
    # 建立分隔線
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    
    # 建立資料列
    data_rows = []
    for row in rows:
        data_rows.append("| " + " | ".join(pad_string(str(item), width) for item, width in zip(row, widths)) + " |")
    
    return "\n".join([header, separator] + data_rows)

# Function to generate customer email
def generate_customer_email(customer, product, date):
    sender = "Alex@aaa.com" if customer == "Alex" else "Ben@bbb.com"
    subject = f"{date.strftime('%Y-%m-%d')} [採購需求] {product['name']}"
    table_data = {
        "columns": ["產品名稱", "數量", "備註"],
        "rows": [[product['name'], product['quantity'], product['remarks']]]
    }
    formatted_table = format_markdown_table(table_data["columns"], table_data["rows"])
    
    body = f"""Hi Jennie,

需購買以下產品，請回覆單號、金額與交期：

{formatted_table}

謝謝！
Best regards,
{customer}"""
    return {
        "received_time": random_timestamp(date),
        "subject": subject,
        "sender": sender,
        "to": "Jennie@vendor.com",
        "body": body,
        "replay_times": 0,
        "attachments": [],
        "tables_json": json.dumps(table_data, ensure_ascii=False)
    }

# Function to generate vendor reply
def generate_vendor_reply(customer_email, product, date):
    product_name = product['name']
    quantity = product['quantity']
    unit_price = pricing[product_name]
    subtotal = quantity * unit_price
    order_number = f"PO-{date.strftime('%Y%m%d')}-{random.randint(100, 999)}"
    delivery_date = (date + timedelta(days=random.randint(5, 10))).strftime('%Y-%m-%d')
    table_data = {
        "columns": ["產品名稱", "數量", "單價 (USD)", "小計 (USD)"],
        "rows": [[product_name, quantity, unit_price, subtotal]]
    }
    formatted_table = format_markdown_table(table_data["columns"], table_data["rows"])
    
    body = f"""Hi {'Alex' if customer_email['sender'] == 'Alex@aaa.com' else 'Ben'},

感謝您的採購需求。以下是訂單詳情：

**訂單單號**：{order_number}
**總金額**：USD {subtotal}
**預計交期**：{delivery_date}

{formatted_table}

如有其他問題，請隨時聯繫。

Best regards,
Jennie
Vendor Co."""
    return {
        "received_time": random_timestamp(date),
        "subject": f"RE: {customer_email['subject']}",
        "sender": "Jennie@vendor.com",
        "to": customer_email['sender'],
        "body": body,
        "replay_times": 1,
        "attachments": [],
        "tables_json": json.dumps(table_data, ensure_ascii=False)
    }

# Generate email data
emails = []
for i in range(10):
    date = random.choice(date_range)
    # Alex's email
    alex_email = generate_customer_email("Alex", alex_products[i], date)
    emails.append(alex_email)
    # Vendor reply to Alex
    emails.append(generate_vendor_reply(alex_email, alex_products[i], date))
    # Ben's email
    ben_email = generate_customer_email("Ben", ben_products[i], date)
    emails.append(ben_email)
    # Vendor reply to Ben
    emails.append(generate_vendor_reply(ben_email, ben_products[i], date))

# 在 df.to_excel 之前加入以下程式碼
# 輸出郵件內容到 txt 檔案
output_file = "email_content.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for i, email in enumerate(emails, 1):
        f.write(f"郵件 #{i}\n")
        f.write("="*50 + "\n")
        f.write(f"收件時間: {email['received_time']}\n")
        f.write(f"主旨: {email['subject']}\n")
        f.write(f"寄件者: {email['sender']}\n")
        f.write(f"收件者: {email['to']}\n")
        f.write(f"內文:\n{email['body']}\n")
        f.write("="*50 + "\n\n")

# Create DataFrame
df = pd.DataFrame(emails)

# Export to Excel
df.to_excel("email_data.xlsx", index=False, engine='openpyxl')

print("Excel file 'email_data.xlsx' has been generated with 40 email records.")
print("Text file 'email_content.txt' has been generated with formatted email content.")