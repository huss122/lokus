
def hälsa(namn, ålder):
    meddelande = f"Hej {namn}, du är {ålder} år."
    return meddelande

for i in range(3):
    print(hälsa("Test", 20 + i))