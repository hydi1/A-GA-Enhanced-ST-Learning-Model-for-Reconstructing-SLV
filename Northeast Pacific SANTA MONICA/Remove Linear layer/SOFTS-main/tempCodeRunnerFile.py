def mean_std(arr):
    arr = np.array(arr, dtype=np.float64)
    return np.nanmean(arr), np.nanstd(arr)


mean_rmse_norm, std_rmse_norm = mean_std(rmse_norms)
mean_mae_norm, std_mae_norm = mean_std(mae_norms)
mean_rmse_unnorm, std_rmse_unnorm = mean_std(rmse_unnorms)
mean_mae_unnorm, std_mae_unnorm = mean_std(mae_unnorms)
mean_r2_batch_norm, std_r2_batch_norm = mean_std(r2_batch_norms)
mean_r2_batch_unnorm, std_r2_batch_unnorm = mean_std(r2_batch_unnorms)
mean_r2_full_unnorm, std_r2_full_unnorm = mean_std(r2_full_unnorms)