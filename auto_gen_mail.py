import pandas as pd
import json
import random
from datetime import datetime, timedelta
import uuid

# Define products for Alex, Ben and Amy
alex_products = [
    {"name": "幻糖", "quantity": 50, "remarks": "七彩繽紛口感"},
    {"name": "甜蜜爆彈", "quantity": 30, "remarks": "爆炸酸甜口感"},
    {"name": "彩虹奇緣", "quantity": 20, "remarks": "多層次水果味"},
    {"name": "嘻哈跳跳糖", "quantity": 15, "remarks": "嘴裡會跳舞"},
    {"name": "月光巧克力球", "quantity": 25, "remarks": "濃郁牛奶巧克力"},
    {"name": "糖雲", "quantity": 10, "remarks": "入口即化"},
    {"name": "魔法能量糖", "quantity": 40, "remarks": "提升專注力"},
    {"name": "莓果派對棒棒糖", "quantity": 60, "remarks": "綜合莓果風味"},
    {"name": "驢子軟糖", "quantity": 12, "remarks": "Q彈有嚼勁"},
    {"name": "甜光迷蹤", "quantity": 8, "remarks": "會發光的糖果"}
]

ben_products = [
    {"name": "幻糖", "quantity": 20, "remarks": "七彩繽紛口感"},
    {"name": "甜蜜爆彈", "quantity": 10, "remarks": "爆炸酸甜口感"},
    {"name": "彩虹奇緣", "quantity": 15, "remarks": "多層次水果味"},
    {"name": "嘻哈跳跳糖", "quantity": 25, "remarks": "嘴裡會跳舞"},
    {"name": "月光巧克力球", "quantity": 12, "remarks": "濃郁牛奶巧克力"},
    {"name": "糖雲", "quantity": 18, "remarks": "入口即化"},
    {"name": "魔法能量糖", "quantity": 30, "remarks": "提升專注力"},
    {"name": "莓果派對棒棒糖", "quantity": 50, "remarks": "綜合莓果風味"},
    {"name": "驢子軟糖", "quantity": 40, "remarks": "Q彈有嚼勁"},
    {"name": "甜光迷蹤", "quantity": 22, "remarks": "會發光的糖果"}
]

amy_products = [
    {"name": "幻糖", "quantity": 35, "remarks": "七彩繽紛口感"},
    {"name": "甜蜜爆彈", "quantity": 25, "remarks": "爆炸酸甜口感"},
    {"name": "彩虹奇緣", "quantity": 30, "remarks": "多層次水果味"},
    {"name": "嘻哈跳跳糖", "quantity": 20, "remarks": "嘴裡會跳舞"},
    {"name": "月光巧克力球", "quantity": 15, "remarks": "濃郁牛奶巧克力"},
    {"name": "糖雲", "quantity": 25, "remarks": "入口即化"},
    {"name": "魔法能量糖", "quantity": 45, "remarks": "提升專注力"},
    {"name": "莓果派對棒棒糖", "quantity": 55, "remarks": "綜合莓果風味"},
    {"name": "驢子軟糖", "quantity": 30, "remarks": "Q彈有嚼勁"},
    {"name": "甜光迷蹤", "quantity": 15, "remarks": "會發光的糖果"}
]

# Pricing for products (for vendor replies)
pricing = {
    "幻糖": 40, "甜蜜爆彈": 15, "彩虹奇緣": 25, "嘻哈跳跳糖": 30,
    "月光巧克力球": 50, "糖雲": 20, "魔法能量糖": 100, "莓果派對棒棒糖": 10,
    "驢子軟糖": 35, "甜光迷蹤": 150
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
    if customer == "Alex":
        sender = "Alex@aaa.com"
    elif customer == "Ben":
        sender = "Ben@bbb.com"
    else:  # Amy
        sender = "Amy@ccc.com"
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

# 修改 generate_vendor_reply 函數中的表格欄位和回覆內容
def generate_vendor_reply(customer_email, product, date):
    product_name = product['name']
    quantity = product['quantity']
    unit_price = pricing[product_name]
    subtotal = quantity * unit_price
    order_number = f"PO-{date.strftime('%Y%m%d')}-{random.randint(100, 999)}"
    delivery_date = (date + timedelta(days=random.randint(5, 10))).strftime('%Y-%m-%d')
    table_data = {
        "columns": ["產品名稱", "數量", "單價 (TWD)", "小計 (TWD)"],
        "rows": [[product_name, quantity, unit_price, subtotal]]
    }
    formatted_table = format_markdown_table(table_data["columns"], table_data["rows"])
    
    if customer_email['sender'] == 'Alex@aaa.com':
        customer_name = 'Alex'
    elif customer_email['sender'] == 'Ben@bbb.com':
        customer_name = 'Ben'
    else:
        customer_name = 'Amy'
    
    body = f"""Hi {customer_name},

感謝您的採購需求。以下是訂單詳情：

**訂單單號**：{order_number}
**總金額**：TWD {subtotal}
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
    emails.append(generate_vendor_reply(alex_email, alex_products[i], date))
    
    # Ben's email
    ben_email = generate_customer_email("Ben", ben_products[i], date)
    emails.append(ben_email)
    emails.append(generate_vendor_reply(ben_email, ben_products[i], date))
    
    # Amy's email
    amy_email = generate_customer_email("Amy", amy_products[i], date)
    emails.append(amy_email)
    emails.append(generate_vendor_reply(amy_email, amy_products[i], date))

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

# 在最後一個 print 語句之前加入以下程式碼
# 準備驗證用的 DataFrame
verify_data = []
for email in emails:
    data = {
        "received_time": email["received_time"],
        "subject": email["subject"],
        "sender": email["sender"],
        "to": email["to"],
        "body": email["body"],
        "replay_times": email["replay_times"]
    }
    
    # 從 tables_json 中提取產品相關資訊
    table_data = json.loads(email["tables_json"])
    if len(table_data["columns"]) == 3:  # 客戶詢價郵件
        data.update({
            "產品名稱": table_data["rows"][0][0],
            "數量": table_data["rows"][0][1],
            "備註": table_data["rows"][0][2],
            "單價": None,
            "小計": None
        })
    else:  # 廠商回覆郵件
        data.update({
            "產品名稱": table_data["rows"][0][0],
            "數量": table_data["rows"][0][1],
            "單價": table_data["rows"][0][2],
            "小計": table_data["rows"][0][3],
            "備註": None
        })
    
    verify_data.append(data)

# 創建驗證用的 DataFrame
df_verify = pd.DataFrame(verify_data)

# 調整欄位順序
columns_order = [
    "received_time", "subject", "sender", "to", 
    "產品名稱", "數量", "單價", "小計", "備註",
    "body", "replay_times"
]
df_verify = df_verify[columns_order]

# 輸出到 Excel
df_verify.to_excel("email_data_verify.xlsx", index=False, engine='openpyxl')

print("Excel file 'email_data.xlsx' has been generated with 40 email records.")
print("Text file 'email_content.txt' has been generated with formatted email content.")
print("Verification Excel file 'email_data_verify.xlsx' has been generated with detailed product information.")