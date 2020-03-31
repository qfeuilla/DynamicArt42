from transfer import Transfer

test = Transfer(20, "./", "./style/style1.jpg", 1, 1e-3, 1e6, 1e10)
# test.load_weight('./model/net_save_epochs_2.pth')
# test.predict("./test/land.jpg")
# test.live_transfer()
test.train()
