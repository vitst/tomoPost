#!/usr/bin/env python3

import os, sys, getopt

listOfAllTools = {}

package_dir = 'tools'
package_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            package_dir)

# import generator classes from generators directory
for filename in os.listdir(package_path):
    modulename, ext = os.path.splitext(filename)
    
    if modulename[0] != '_' and ext == '.py':
        subpackage = '{0}.{1}'.format(package_dir, modulename)
        obj = getattr(__import__(subpackage, globals(), locals(), [modulename]),
                      modulename)
        listOfAllTools.update({obj().__toolName__: obj})

short_description = "\nThis is a collection of tools to postprocess X-ray " \
                    "tomography data."

description = '''
  {0:s}\n
  
  
  
  Usage:

    name of the tool(tool: *name*)
    list of parameters (*name*: *value*)
    .....

  To list all available tools:
    `./runTomo -l` or `./runTomo --list`

  To see parameters for correctTilt tool:
    `./runTomo -i correctTilt`

'''.format(short_description)


package_name = "X-ray tomography postprocessing"
__doc__ = description
__version__ = 0.0
__author__ = 'Vitaliy Starchenko'

def help(short):
    if short:
        print(short_description)
    else:
        print(description)
    print(
        'options:\n'
        '\t-h\t\t\t\tshort help\n'
        '\t--help\t\t\t\textended help\n'
        '\t-l [or --list]\t\t\tlist of all tools\n'
        '\t-i [or --info] <toolName>\tprint description of all parameters '
        'for the specified tool\n'
        '\t-c [or --check] <fileName>\tcheck parameters in '
        'the tool dictionary file\n'
        '\t-g <toolName>\tgenerates a `toolDict` dictionary file using toolName\n'
        '\t-d <dictioanryFileName>\texecutes a tool using a dictionary file\n'
        
        'examples:\n'
        '\t./runTomo -d <dictioanryFileName>\n'
    )

def errorMsg(msg):
    return '\n*** Error. There is no tool `{0:s}`.\n' \
           'Run `./runTomo -l` or `./runTomo --list` to see the list of ' \
           'all available tools.\n'.format(msg)

def generateDictionary(toolName):
  if os.path.isfile('toolDict'):
    choice = ''
    while choice!='y' and choice!='n':
      # need to know what is the version of the interpreter
      if sys.version_info[0]==2:
        choice=raw_input('\nThe file `toolDict` exists already. '
                         'Do you want to delete the file first? (y/n): ')
      else:
        choice=input('\nThe file `toolDict` exists already. '
                     'Do you want to delete the file first? (y/n): ')
    if choice=='y':
      os.remove('toolDict')
    else:
      return

  try:
    toolClass = listOfAllTools[toolName]
  except KeyError:
    print(errorMsg(toolName))
    sys.exit(2)

  tool = toolClass()
  tool.generate()


def check_parameters(dictName):
  try:
    f = open(dictName, 'r')
  except IOError:
    print('\n*** Error. Cannot open `{0:s}`\n'.format(dictName))
  else:
    toolNameLine = f.readline()
    f.close()
    if 'toolName:' in toolNameLine:
      toolName = toolNameLine.split()[1]
      try:
        toolClass = listOfAllTools[toolName]
      except KeyError:
        print(errorMsg(toolName))
        sys.exit(2)

      tool = toolClass()
      if not tool.run_check(dictName):
        sys.exit(2)
    else:
      print('The name of the tool was not found in `{0:s}` dictionary.\n'
            'You can always generate a template dictionary file by running:\n'
            ' ./runTomo -g <toolName>'.format(dictName))
  return

 

def callTool(arg):
    try:
        f = open(arg, 'r')
    except IOError:
        print('\n*** Error. Can not open `{0:s}`\n'.format(arg))
    else:
        toolLine = f.readline()
        f.close()
        if 'toolName:' in toolLine:
            toolName = toolLine.split()[1]
            
    try:
        toolClass = listOfAllTools[toolName]
    except KeyError:
        print(errorMsg(toolName))
        sys.exit(2)
    
    tool = toolClass()
    tool.execute(arg)
    
    print('\nThe tool {0:s} is done.\n'.format(toolName))
    return


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hvli:c:g:d:",
                                   ["help", "version", "list",
                                    "info=", "check="])
    except getopt.GetoptError:
        print('\n*** Error. Options are not correct.')
        help(True)
        sys.exit(2)
    
    if len(opts) == 0:
        print('for help run -h option')
    
    for opt, arg in opts:
        if opt == "-h":
            help(True)
        elif opt == "--help":
            help(False)
        elif opt in ("-v", "--version"):
            print("{0:s}. Version {1:g}".format(package_name, __version__))
        elif opt in ("-l", "--list"):
            print('\nList of available tools: ')
            for key in sorted(listOfAllTools.keys()):
                print('  {0:s}\n'.format(key))
        elif opt in ("-i", "--info"):
            try:
                toolClass = listOfAllTools[arg]
            except KeyError:
                print(errorMsg(arg))
                sys.exit(2)
            tool = toolClass()
            tool.info()
        elif opt in ("-c", "--check"):
            check_parameters(arg)
        elif opt == "-g":
            generateDictionary(arg)
        elif opt in ("-d"):
            check_parameters(arg)
            callTool(arg)

if __name__ == "__main__":
    main(sys.argv[1:])
