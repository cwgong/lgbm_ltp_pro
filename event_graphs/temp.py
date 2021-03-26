from LAC import LAC
from ddparser import DDParser

def test_lac():
    lac = LAC(mode="seg")

    text = "人民银行认真贯彻党中央、国务院关于“六稳”“六保”工作的决策部署"
    seg_result = lac.run(text)
    print(seg_result)

def test_ddp():
    ddp = DDParser()
    print(ddp.parse("百度是一家高科技公司"))


if __name__ == "__main__":
    test_lac()
