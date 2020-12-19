from src.eval import f1_score

#test basic case
def test_f1_score0():
    truth = [(1,0),(2,1),(3,1)]
    proposed = [(1,1),(2,1),(3,1)]
    f1 = f1_score(proposed, truth)
    # D00 = 0, D10 = 0, D01 = 2, D11 = 1
    # P = 1/3, R = 1, F1 = 2/3 / 4/3 = 1/2
    assert(f1 == 1/2)

#test that mismatched values are ignored
def test_f1_score1():
    truth = [(1,0),(2,1),(3,1)]
    proposed = [(1,1),(2,1),(3,1)]
    truth.append((4,2))
    f1 = f1_score(proposed, truth)
    assert(f1 == 1/2)

    proposed.append((5,3))
    f1 = f1_score(proposed, truth)
    assert(f1 == 1/2)

#test edge case of incalculable precision or recall
def test_f1_score2():
    truth = [(1, 0), (2, 0)]
    proposed = [(1, 0), (2, 1)]
    f1 = f1_score(proposed, truth)
    # D00 = 0, D10 = 1, D01 = 0, D11 = 0
    # P = N/A, R = 0, F1 = N/A
    assert (f1 == False)

#test more complex f1
def test_f1_score3():
    truth = [(1, 1), (2, 1), (3, 1), (4, 3), (5, 4)]
    proposed = [(1, 0), (2, 1), (3, 1), (4, 3), (5, 3)]
    f1 = f1_score(proposed, truth)
    # D00 = 6, D10 = 2, D01 = 1, D11 = 1
    # P = 1/2, R = 1/3, F1 = 1/3 / 5/6 = .4
    assert (f1 == .4)


test_f1_score0()
test_f1_score1()
test_f1_score2()
test_f1_score3()