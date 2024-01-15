import xml.etree.ElementTree as ET

def main():
    tree = ET.parse('/example/corpus/fulltext/06_1.xml')
    root = tree.getroot()

if __name__ == '__main__':
    main()