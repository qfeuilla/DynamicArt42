from transfer import Transfer

test = Transfer(20, "./", "./style/style1.jpg", 4, 1e-3, 1, 5)
# test.load_weight('./model/starry-night.pth')
# test.predict("./test/land.jpg")
test.train()
