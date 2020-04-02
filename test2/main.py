from transfer import Transfer

test = Transfer(20, "./", "./style/style1.jpg", 4, 1e-3, 1e4, 1e10)
# test.load_weight('./model/net_save_epochs_2.pth')
# test.predict("./test/emma.jpg")
# test.live_transfer()
test.train()
