def cycle_dataloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
