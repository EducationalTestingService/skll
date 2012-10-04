import math
import texttable

# a utility method to convert floats to strings of fixed length
make_printable = lambda f: '{:<05}'.format(f)
make_printable_prf = lambda f: '  -' if math.isnan(f) else '{:>3}'.format(int(f))


# a function to format the appropriate number(s) for printing in the tables
# this also returns the size of the column in characters
def table_number_formatter(resultlist):
    accuracy = resultlist[0]
    out = [make_printable(accuracy)]
    class_accuracies = resultlist[1:3] if len(resultlist) == 9 else resultlist[1:4]
    out.append("  ".join(map(make_printable, class_accuracies)))
    prfs = resultlist[3:] if len(resultlist) == 9 else resultlist[4:]
    prfstr = "  ".join([" ".join(map(make_printable_prf, prfs[i: i + 3])) for i in range(0, 9, 3)])
    prfstr = prfstr.rstrip()
    out.append(prfstr)
    return out


# a function to generate ASCII tables with the various results
def generate_ascii_tables(given_lexicons, given_featuresets, given_classifiers, resultshash):

    # initialize ascii results tables
    acctable = texttable.Texttable()
    cacctable = texttable.Texttable()
    prftable = texttable.Texttable()

    # set up the tables headers and alignment
    header = ["Classifier", "FeatSet"]
    alignment = ["l", "l"]
    valignment = ["m", "m"]
    for lexicon in given_lexicons:
        header.append(lexicon)
        alignment.append("l")
        valignment.append("m")
    acctable.set_cols_align(alignment)
    acctable.set_cols_valign(valignment)

    # set up the tables headers and alignment
    alignment = ["l", "l"]
    valignment = ["m", "m"]
    for lexicon in given_lexicons:
        alignment.append("l")
        valignment.append("m")
    cacctable.set_cols_align(alignment)
    cacctable.set_cols_valign(valignment)

    # set up the tables headers and alignment
    alignment = ["l", "l"]
    valignment = ["m", "m"]
    for lexicon in given_lexicons:
        alignment.append("l")
        valignment.append("m")
    prftable.set_cols_align(alignment)
    prftable.set_cols_valign(valignment)

    # start filling up the tables' contents
    acc_table_contents = [header]
    cacc_table_contents = [header]
    prf_table_contents = [header]

    arow = []
    carow = []
    prfrow = []
    for classifier in given_classifiers:
        arow.append(classifier)
        arow.append("\n".join(given_featuresets))
        carow.append(classifier)
        carow.append("\n".join(given_featuresets))
        prfrow.append(classifier)
        prfrow.append("\n".join(given_featuresets))
        for lexicon in given_lexicons:
            accs = []
            caccs = []
            prfs = []
            for featureset, resultlist in resultshash[classifier][lexicon]:
                numbers = table_number_formatter(resultlist)
                accs.append(numbers[0])
                caccs.append(numbers[1])
                prfs.append(numbers[2])
            arow.append("\n".join(accs))
            carow.append("\n".join(caccs))
            prfrow.append("\n".join(prfs))
        acc_table_contents.append(arow)
        cacc_table_contents.append(carow)
        prf_table_contents.append(prfrow)
        arow = []
        carow = []
        prfrow = []

    # set the appropriate column size for the class accuracy table
    # depending on whether we are doing sentiment or polarity
    max_featset_col_size = max(map(len, given_featuresets))
    colwidth = len(cacc_table_contents[1][2].split('\n')[0])
    if colwidth < 19:
        width_array = [10, max_featset_col_size] + [14] * len(given_lexicons)
        cacctable.set_cols_width(width_array)
    else:
        width_array = [10, max_featset_col_size] + [19] * len(given_lexicons)
        cacctable.set_cols_width(width_array)

    # set the appropriate column size for the PRF table
    # depending on whether we are doing sentiment or polarity
    colwidth = len(prf_table_contents[1][2].split('\n')[0])
    if colwidth < 27:
        width_array = [10, max_featset_col_size] + [25] * len(given_lexicons)
        prftable.set_cols_width(width_array)
    else:
        width_array = [10, max_featset_col_size] + [37] * len(given_lexicons)
        prftable.set_cols_width(width_array)

    # add the respective contents to each table
    acctable.add_rows(acc_table_contents)
    cacctable.add_rows(cacc_table_contents)
    prftable.add_rows(prf_table_contents)

    return acctable, cacctable, prftable


def generate_tsv_rows(test_set_name, given_lexicons, given_featuresets, given_classifiers, resultshash):
    tsvrows = []
    for classifier in given_classifiers:
        for lexicon in given_lexicons:
            for featureset, results in resultshash[classifier][lexicon]:
                row = [test_set_name, lexicon, featureset, classifier]
                row.extend(results)
                tsvrows.append(row)
    return tsvrows
