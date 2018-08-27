# IPython log file

import sys
from abc import ABCMeta, abstractmethod


class AbstractBaseTool(object):
    '''
    This is the base class for tomography postprocessing tools
    '''
    __metaclass__ = ABCMeta
    
    #blockMeshFileName = 'blockMeshDict'
    
    dictFileName = 'toolDict'
    
    def __init__(self):
        pass
    
    def info(self):
        '''
        Prints info about the variables needed for particular tool
        '''
        prntCont = '\nDescription of the tool:\n'
        prntCont += self.__doc__
        prntCont += '\nList of parameters in the dictionary:\n'
        for prm in self.parameters:
            prntCont += '\t' + prm[0] + ':\t' + \
                        str(prm[1]) + \
                        '\t' + prm[2] + '\n'
        print(prntCont)
        return

    ############################################################################
    # read/write
    ############################################################################
    def read_dict(self, dictFileName):
        '''
        Read a file and return string lines
        '''
        lines = ''
        try:
            f = open(dictFileName, 'r')
        except IOError:
            print('\nError. Cannot open `%s`' % dictFileName)
            sys.exit(2)
        else:
            lines = f.readlines()
            f.close()
        return lines

    ############################################################################

    def check_parameters(self, lines):
        '''
        Check whether parameters in dictionary are OK
        '''
        check = True
    
        loc_tool = [line for line in lines if 'toolName' in line]
        if len(loc_tool) == 0:
            check = False
            print(
                '\n*** Error. There is no tool name '
                'in the dictionary file\n'
            )
        elif len(loc_tool) > 1:
            check = False
            print(
                '\n*** Error. There more than one tool name'
                ' in the dictionary file\n'
            )
        else:
            toolName = loc_tool[0].split()[1]
        
            if toolName != self.__toolName__:
                check = False
                print('\n*** Error. The tool name `{0:s}`'
                      'does not correspond to the name from '
                      'the dictionary `{1:s}`\n'.
                      format(self.__toolName__, toolName))
    
        for prm in self.parameters:
            loc_check, val_par, description = self.check_a_parameter(prm[0], lines)
            check = loc_check and check
            print("{0:15s}  {1:15}  {2:15}".format(prm[0], val_par, description))
    
        return check, self.__toolName__

    def run_check(self, dictFileName):
        lines = self.read_dict(dictFileName)
        result, name = self.check_parameters(lines)
        if result:
            print(
                '\nThe parameters in the dictionary `{0:s}`'
                'for the tool `{1:s}` are OK.\n'.
                    format(dictFileName, name)
            )
        else:
            print(
                '\nThere is something wrong in the dictionary `{0:s}` '
                'for the tool `{1:s}`.\n'
                'See comments above.\n'.format(dictFileName, name)
            )
    
        return result

    def check_a_parameter(self, parameterName, lines):
        '''
        General function to check the format of a parameter in dictionary
        '''
    
        check = True
        val_param = None
        desc_param = None
        if not parameterName in [row[0] for row in self.parameters]:
            print('\n*** Error. There is no parameter '
                  '`{0:s}`\n'.format(parameterName))
            sys.exit(2)
        else:
            par_type = type(self.get_parameter(parameterName))
        
            loc_param = [line for line in lines if parameterName in line]
            if len(loc_param) == 0:
                check = False
                print(
                    '\n*** Error. There is no `{0:s}` in '
                    'the dictionary file\n'.format(parameterName)
                )
            elif len(loc_param) > 1:
                check = False
                print(
                    '\n*** Error. There more than one `{0:s}` in '
                    'the dictionary file\n'.format(parameterName)
                )
            else:
                try:
                    val_param = par_type(loc_param[0].split()[1])
                    desc_param = loc_param[0].split()[2:]
                    desc_param = ''.join('{} '.format(e) for e in desc_param)
                    #desc_param = ''.join(str(e) for e in desc_param)
                    #desc_param = ''.join(map(str, desc_param))
                    #desc_param = ''.join(desc_param1)
                    self.set_parameter(parameterName, val_param)
                except IndexError:
                    check = False
                    print(
                        '\n*** Error. `{0:s}` does not have '
                        'a value.'.format(parameterName)
                    )
                except ValueError:
                    check = False
                    print(
                        '\n*** Error. `{0:s}` has a wrong number '
                        'format.'.format(parameterName)
                    )
    
        return check, val_param, desc_param

    def get_parameter(self, parameterName):
        ind = [row[0] for row in self.parameters].index(parameterName)
        return self.parameters[ind][1]

    def set_parameter(self, parameterName, value):
        ind = [row[0] for row in self.parameters].index(parameterName)
        self.parameters[ind][1] = value
        return

    def generate(self):
        '''
        Create a tool dictionary
        '''
        file = open(self.dictFileName, 'w')
        fileCont = 'toolName: {0:s}\n'.format(self.__toolName__)
        for prm in self.parameters:
            fileCont += prm[0] + ':\t' + str(prm[1]) + '\t' + prm[2] + '\n'
        file.write(fileCont)
        file.close()
        
        print('Dictionary file `{0:s}` was generated.'.
              format(self.dictFileName))
    
        return

    ############################################################################
    # abstract
    ############################################################################

    @abstractmethod
    def execute(self, dictFileName):
        '''
        Creates a blockMeshDict. It should return True or False depending on the operation,
        whether it was successful of not.
        '''
        pass


