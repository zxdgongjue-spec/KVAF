BEGIN {
    if (candidates == "") {
        exit 2
    }

    candidate_count = split(candidates, wanted, /[[:space:]]+/)
    for (candidate_index = 1;
         candidate_index <= candidate_count;
         candidate_index++) {
        if (wanted[candidate_index] != "") {
            selected[wanted[candidate_index]] = 1
        }
    }
}

{
    symbol = $1
    base = symbol
    sub(/\..*$/, "", base)
    if (selected[base] && !seen[base]++) {
        print symbol
    }
}
