import csv  

async def initial_csv(data):
    with open('data.csv', 'a+', newline="", encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(data)



