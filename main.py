import nni
import logging
import sys
from utils import *

logger = logging.getLogger("price_volatility")

if "__main__" == __name__:
    try:
        # 1: linear, 2: mc dropout, 3: predict plus or minus
        model_num = int(sys.argv[1])
        is_denoise = int(sys.argv[2])
        epochs = int(sys.argv[3])

        RCV_CONFIG = nni.get_next_parameter()
        lr = RCV_CONFIG['lr']
        weight_decay = RCV_CONFIG['weight_decay']
        if model_num == 2:
            dropout_rate = RCV_CONFIG['dropout_rate']
        else:
            dropout_rate = 0

        # change data to train, test
        train_data, train_label, test_data, test_label, check_data = get_data(is_denoise, model_num)

        # get train model
        model = get_model(RCV_CONFIG['param'], RCV_CONFIG['num_hidden'], dropout_rate, model_num)
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


        final = None
        test_loss_min = 100
        for e in range(epochs):
            train_loss = 0
            test_loss = 0
            #train
            for inputs, label in zip(train_data, train_label):
                if model_num == 2:
                    mu, sig = model(inputs)
                    loss = loss_fn(label, mu, sig)
                elif model_num == 3:
                    model.train()
                    pm = model(inputs, model_num)
                    loss = nn.CrossEntropyLoss()(pm, label[0].type(torch.int64))
                else:
                    model.train()
                    output = model(inputs)
                    loss = nn.SmoothL1Loss()(label, output)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            train_loss /= len(train_data)
            # nni.report_intermediate_result(train_loss)

            # test
            with torch.no_grad():
                for test_in, test_la in zip(test_data, test_label):
                    if model_num == 2:
                        mu, sig = model(test_in)
                        loss = loss_fn(test_la, mu, sig)
                    elif model_num == 3:
                        model.eval()
                        pm = model(test_in, model_num)
                        loss = nn.CrossEntropyLoss()(pm, test_la[0].type(torch.int64))
                    else:
                        model.eval()
                        output = model(test_in)
                        loss = nn.SmoothL1Loss()(test_la, output)
                    test_loss += loss.item()
            test_loss /= len(test_data)
            nni.report_intermediate_result(test_loss)
            if test_loss < test_loss_min:
                test_loss_min = test_loss
        nni.report_final_result(test_loss_min)
        check_model(model, check_data, test_data, test_label, model_num)

    except Exception as e:
        logger.exception(e)
        raise e