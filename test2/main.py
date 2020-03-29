from transfer import Transfer

test = Transfer(20, "./", "./style/mosaic.jpg", 1e-3, 1, 5)
test.load_weight('./model/save_style1.pth')
test.predict("./test/land.jpg")
test.train()
