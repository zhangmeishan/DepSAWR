import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--refer', default='default.cfg')
    argparser.add_argument('--template', default='default.cfg')
    argparser.add_argument('--output', default='default.cfg')

    args, extra_args = argparser.parse_known_args()

    items = {}
    with open(args.refer, 'r', encoding='utf-8') as infile:
        for line in infile:
            divides = line.strip().split(' = ')
            new_divides = line.strip().split('=')
            section_num, new_section_num = len(divides), len(new_divides)

            if section_num != new_section_num or section_num > 2:
                print(line)

            if section_num == 2:
                items[divides[0].strip()] = divides[1].strip()

    output = open(args.output, 'w', encoding='utf-8')

    with open(args.template, 'r', encoding='utf-8') as infile:
        for line in infile:
            divides = line.strip().split(' = ')
            new_divides = line.strip().split('=')
            section_num, new_section_num = len(divides), len(new_divides)

            if section_num != new_section_num or section_num > 2:
                print(line)

            if section_num < 2:
                output.write(line)
            else:
                key_str = divides[0].strip()
                value_str = divides[1].strip()

                if key_str in items:
                    value_str = items[key_str]

                output.write(key_str + " = " + value_str + "\n")

    output.close()
