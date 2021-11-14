import csv  

async def initial_csv(data):
    with open('data.csv', 'a+', newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(data)



