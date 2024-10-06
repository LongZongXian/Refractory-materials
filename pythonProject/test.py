import torch

def test_regressor(model, criterion, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        for idx,(img_data, table_data, targets) in enumerate(test_dataloader):
            img_data = img_data.to(device)
            table_data = table_data.to(device)
            targets = targets.to(device)

            outputs = model(img_data, table_data)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * img_data.size(0)

        test_loss = running_loss / len(test_dataloader.dataset)

        return test_loss
        # print(f'Test_Loss: {test_loss:.4f}')
        # writerTest.add_scalar("Loss/Test", test_loss, epoch_num)


