import time
from tqdm import tqdm
import numpy as np

from common.logger import Logger


class ds:
    def __init__(self):
        self.output = []
        self.label = []


def mean_average_precision_normal(database_output, database_labels, query_output, query_labels, R, verbose=0):
    query_num = query_output.shape[0]

    sim = np.dot(database_output, query_output.T)
    start_time = time.time()
    ids = np.argsort(-sim, axis=0)
    end_time = time.time()
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, end_time - start_time))
    APx = []
    precX = []
    recX = []
    label_matchs = calc_label_match_matrix(database_labels, query_labels)
    for i in range(query_num):
        if i % 100 == 0:
            tmp_time = time.time()
            print("query map {:d}, time: {:.3f}".format(i, tmp_time - end_time))
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)

        all_sim_num = label_matchs.all_sims[i]
        recX.append(float(relevant_num) / all_sim_num)
        precX.append(float(relevant_num) / R)

        Lx = np.cumsum(imatch)

        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if verbose > 1:
            print(relevant_num, relevant_num, APx[-1])
    if verbose > 0:
        print("MAP: %f" % np.mean(np.array(APx)))
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), np.mean(np.array(APx), 0)


def mean_average_precision_normal_optimized_label(database_output, database_labels, query_output, query_labels,
                                                  R, verbose=0, label_matchs=None):

    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0

    label_matrix_time = -1
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)
        label_matrix_time = time.time() - tmp_time
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))

    query_num = query_output.shape[0]

    sim = np.dot(database_output, query_output.T)
    start_time = time.time()
    ids = np.argsort(-sim, axis=0)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))
    APx = []
    precX = []
    recX = []

    for i in range(query_num):
        if i % 100 == 0:
            tmp_time = time.time()
            print("query map {:d}, time: {:.3f}".format(i, tmp_time - end_time))
            end_time = tmp_time
        idx = ids[:, i]
        imatch = label_matchs.label_match_matrix[i, idx[0:R]]
        relevant_num = np.sum(imatch)

        all_sim_num = label_matchs.all_sims[i]
        recX.append(float(relevant_num) / all_sim_num)
        precX.append(float(relevant_num) / R)

        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)
        if relevant_num > 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if verbose > 1:
            print(relevant_num, relevant_num, APx[-1])
    if verbose > 0:
        print("MAP: %f" % np.mean(np.array(APx)))
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))
    print("total time(no label matrix): {:.3f}".format(time.time() - start_time))
    if label_matrix_time > 0:
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))
    return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), np.mean(np.array(APx), 0)


def partition_arg_topK(matrix, K, axis=0):

    a_part = np.argpartition(matrix, K, axis=axis)
    if axis == 0:
        row_index = np.arange(matrix.shape[1 - axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1 - axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


def mean_average_precision_normal_optimized_topK(database_output, database_labels, query_output, query_labels, R,
                                                 verbose=0, label_matchs=None):

    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0

    label_matrix_time = -1
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)
        label_matrix_time = time.time() - tmp_time
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))

    query_num = query_output.shape[0]

    sim = -np.dot(query_output, database_output.T)
    start_time = time.time()
    topk_ids = partition_arg_topK(sim, R, axis=1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))


    column_index = np.arange(query_num)[:, None]
    imatchs = label_matchs.label_match_matrix[column_index, topk_ids]
    relevant_nums = np.sum(imatchs, axis=1)

    recX = relevant_nums.astype(float) / label_matchs.all_sims
    precX = relevant_nums.astype(float) / R

    Lxs = np.cumsum(imatchs, axis=1)
    Pxs = Lxs.astype(float) / np.arange(1, R + 1, 1)
    APxs = np.sum(Pxs * imatchs, axis=1)[relevant_nums > 0] / relevant_nums[relevant_nums > 0]
    meanAPxs = np.sum(APxs) / query_num
    if verbose > 0:
        print("MAP: %f" % meanAPxs)
    print("total query: {:d}, sorting time: {:.3f}".format(query_num, sort_time))
    print("total time(no label matrix): {:.3f}".format(time.time() - start_time))
    if label_matrix_time > 0:
        print("calc label matrix: time: {:.3f}".format(label_matrix_time))
    return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), meanAPxs


def get_precision_recall_by_Hamming_Radius_All(database_output, database_labels, query_output, query_labels):
    signed_query_output = np.sign(query_output)
    signed_database_output = np.sign(database_output)

    bit_n = signed_query_output.shape[1]

    ips = np.dot(signed_query_output, signed_database_output.T)
    ips = (bit_n - ips) / 2
    precX = np.zeros((ips.shape[0], bit_n + 1))
    recX = np.zeros((ips.shape[0], bit_n + 1))
    mAPX = np.zeros((ips.shape[0], bit_n + 1))

    start_time = time.time()
    ids = np.argsort(ips, 1)
    end_time = time.time()
    print("total query: {:d}, sorting {:.3f}".format(ips.shape[0], end_time - start_time))
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            print("query map {:d}, {:.3f}".format(i, tmp_time - end_time))
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1

        idx = ids[i, :]
        imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
        all_sim_num = int(np.sum(imatch))

        counts = np.bincount(ips[i, :].astype(np.int64))

        for r in range(bit_n + 1):
            if r >= len(counts):
                precX[i, r] = precX[i, r - 1]
                recX[i, r] = recX[i, r - 1]
                mAPX[i, r] = mAPX[i, r - 1]
                continue

            all_num = int(np.sum(counts[0:r + 1]))

            if all_num != 0:
                match_num = np.sum(imatch[0:all_num])
                precX[i, r] = np.float(match_num) / all_num
                recX[i, r] = np.float(match_num) / all_sim_num

                rel = match_num
                Lx = np.cumsum(imatch[0:all_num])
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX[i, r] = np.sum(Px * imatch[0:all_num]) / rel
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX), 0), np.mean(np.array(recX), 0), np.mean(np.array(mAPX), 0)


def get_precision_recall_by_Hamming_Radius(database_output, database_labels, query_output, query_labels, radius=2):
    signed_query_output = np.sign(query_output)
    signed_database_output = np.sign(database_output)

    bit_n = signed_query_output.shape[1]

    ips = np.dot(signed_query_output, signed_database_output.T)
    ips = (bit_n - ips) / 2

    start_time = time.time()
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))

    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    zero_count = 0
    for i in range(ips.shape[0]):
        if i % 100 == 0:
            tmp_time = time.time()
            end_time = tmp_time
        label = query_labels[i, :]
        label[label == 0] = -1
        idx = np.reshape(np.argwhere(ips[i, :] <= radius), (-1))
        all_num = len(idx)
        if all_num != 0:
            imatch = np.sum(database_labels[idx[:], :] == label, 1) > 0
            match_num = np.sum(imatch)
            precX.append(np.float(match_num) / all_num)
            matchX.append(match_num)
            allX.append(all_num)
            all_sim_num = np.sum(
                np.sum(database_labels[:, :] == label, 1) > 0)
            recX.append(np.float(match_num) / all_sim_num)
            if radius < 10:
                ips_trad = np.dot(
                    query_output[i, :], database_output[ids[i, 0:all_num], :].T)
                ids_trad = np.argsort(-ips_trad, axis=0)
                db_labels = database_labels[ids[i, 0:all_num], :]

                rel = match_num
                imatch = np.sum(db_labels[ids_trad, :] == label, 1) > 0
                Lx = np.cumsum(imatch)
                Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
                if rel != 0:
                    mAPX.append(np.sum(Px * imatch) / rel)
            else:
                mAPX.append(np.float(match_num) / all_num)

        else:
            print('zero: %d, no return' % zero_count)
            zero_count += 1
            precX.append(np.float(0.0))
            recX.append(np.float(0.0))
            mAPX.append(np.float(0.0))
            matchX.append(0.0)
            allX.append(0.0)
    print("total query: {:d}, sorting time: {:.3f}".format(ips.shape[0], sort_time))
    print("total time: {:.3f}".format(time.time() - start_time))
    return np.mean(np.array(precX)), np.mean(np.array(recX)), np.mean(np.array(mAPX))


class LabelMatchs(object):
    def __init__(self, label_match_matrix):
        self.label_match_matrix = label_match_matrix
        self.all_sims = np.sum(label_match_matrix, axis=1)


def calc_label_match_matrix(database_labels, query_labels):

    return LabelMatchs(np.dot(query_labels, database_labels.T) > 0)


def mean_average_precision(database_hash, test_hash, database_labels, test_labels, option):  
    R = option.R
    T = option.T
    database_hash[database_hash < T] = -1
    database_hash[database_hash >= T] = 1
    test_hash[test_hash < T] = -1
    test_hash[test_hash >= T] = 1

    query_num = test_hash.shape[0]  
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)
    del sim
    APx = []
    Recall = []
    Pre = []
    iteration = tqdm(range(query_num), desc="CalMAP")
    for i in iteration:  
        label = test_labels[i, :]  
        if np.sum(label) == 0:  
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)  
        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  
            APx.append(0)
        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / float(all_num)
        Pre.append(relevant_num / R)
        Recall.append(r)
    return np.mean(np.array(APx)), np.mean(np.array(Recall)), np.mean(np.array(Pre))


def get_precision_recall_by_Hamming_Radius_optimized(database_output, database_labels, query_output, query_labels,
                                                     radius=2, label_matchs=None, coarse_sign=True, fine_sign=False):

    query_labels[query_labels < 0] = 0
    database_labels[database_labels < 0] = 0
    bit_n = query_output.shape[1]  
    coarse_query_output = np.sign(query_output)
    coarse_database_output = np.sign(database_output)
    del query_output, database_output
    fine_query_output = coarse_query_output
    fine_database_output = coarse_database_output

    label_matrix_time = -1
    Logger.info("calculate match matrix")
    if label_matchs is None:
        tmp_time = time.time()
        label_matchs = calc_label_match_matrix(database_labels, query_labels)
        label_matrix_time = time.time() - tmp_time
        Logger.info("calc label matrix: time: {:.3f}\n".format(label_matrix_time))
    start_time = time.time()
    ips = np.dot(coarse_query_output, coarse_database_output.T)
    ips = (bit_n - ips) / 2
    ids = np.argsort(ips, 1)
    end_time = time.time()
    sort_time = end_time - start_time
    Logger.info("total query: {:d}, sorting time: {:.3f}\n".format(ips.shape[0], sort_time))
    all_nums = np.sum(ips <= radius, axis=1)
    del ips
    precX = []
    recX = []
    mAPX = []
    matchX = []
    allX = []
    iteration = tqdm(range(coarse_query_output.shape[0]), desc="CalMAP")
    for i in iteration:
        all_num = all_nums[i]

        if all_num != 0:
            idx = ids[i, 0:all_num]
            if fine_sign:
                imatch = label_matchs.label_match_matrix[i, idx[:]]
            else:
                ips_continue = np.dot(fine_query_output[i, :], fine_database_output[idx, :].T)
                subset_idx = np.argsort(-ips_continue, axis=0)
                idx_continue = idx[subset_idx]
                imatch = label_matchs.label_match_matrix[i, idx_continue]

            match_num = int(np.sum(imatch))
            matchX.append(match_num)
            allX.append(all_num)
            precX.append(float(match_num) / all_num)
            all_sim_num = label_matchs.all_sims[i]
            recX.append(float(match_num) / all_sim_num)

            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, all_num + 1, 1)
            if match_num != 0:
                mAPX.append(np.sum(Px * imatch) / match_num)
            else:
                mAPX.append(0)

    if label_matrix_time > 0:
        pass
    meanPrecX = 0 if len(precX) == 0 else np.mean(np.array(precX))
    meanRecX = 0 if len(recX) == 0 else np.mean(np.array(recX))
    meanMAPX = 0 if len(mAPX) == 0 else np.mean(np.array(mAPX))
    del fine_database_output, fine_query_output, label_matchs
    return meanPrecX, meanRecX, meanMAPX
