import DQAS01
import DQAS02
import DQAS03
import DQAS04

if __name__ == '__main__':
    DQAS01.main(epochs=50, threshold=20, scheme_epochs=5)
    for i in range(5):
        DQAS02.main(epochs=20, limit=30, scheme_epochs=3)
        DQAS03.main(epochs=20, limit=30, scheme_epochs=3)
    DQAS04.main(epochs=200)
