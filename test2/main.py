from transfer import Transfer

test = Transfer(20, "./", "./style/style1.jpg", 2, 1e-3, 1, 5, 1e-2)
# test.load_weight('./model/save_style1.pth')
# test.predict("./test/land.jpg")
test.train()
