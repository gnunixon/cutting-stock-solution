from itertools import product
import numpy as np
from copy import deepcopy
from scipy import signal

COLUMNS = 5
ROWS = 102


def find_empty_cel(matrix):
    """
    Give a matrix and find from top-left corner to bottom-right corner the
    first cell equal with 1 (meen empty). Returns the coordinates of cell.

    :param list matrix: The matrix with 1 for empty cell and -1 for full
    :returns: Coordinates of first empty cell (row, column)
    :rtype: tuple
    """
    rows, columns = np.where(matrix == 1)
    return int(rows[0]), int(columns[0])


def validate_area(dimensions, columns, rows):
    """
    Verify if the given list of articles have an area less or equal with the
    area of page

    :param list dimensions: the list of dimensions of articles
    :returns: True if the area of all articles is less or equal with the page
    :rtype: bool
    """
    area = sum([item['area'] for item in dimensions])
    page_area = columns * rows
    if (area <= page_area):
        return True
    return False


def add_area(dimensions):
    """
    Add to the dict with the dimensions of articles the sum of their areas

    :param list dimensions: the list of dimensions of articles
    :returns: the list with area added
    :rtype: list
    """
    area = sum([item['area'] for item in dimensions])
    ret = {'area': area, 'dimensions': dimensions}
    return ret


def validate_step(row, col, rows, cols, matrix):
    """
    Verify if we can put the article at the given coordinates without
    intersectation with other article

    :param int row: the y-coordinate of top-left corner of article
    :param int col: the x-coordinate of top-left corner of article
    :param int rows: how many rows have the height of article
    :param int cols: how may columns have the width of article
    :param list matrix: the current layout of page
    :returns: True if we have enough space at the given coords
    :rtype: bool
    """
    free_cels = np.count_nonzero(matrix[row:row + rows, col:col + cols] == 1)
    if free_cels != rows * cols:
        return False
    return True


def create_matrix(rows, columns):
    """
    Create a 1-filled rows x columns matrix

    :param int rows: Number of rows in matrix
    :param int columns: Number of columns in matrix
    :returns: The resulted 1-filled matrix with dimensions
    :rtype: list
    """
    return np.ones((rows, columns), dtype=int)


def generate_layouts(articles):
    """
    This function will generate a list of potential articles for page. Each
    article in our case can be layouted in many modes (on 1..5 columns, with or
    without photo and so on). We need to generate all possible combinations of
    this layouts and sort them by area.

    :param list articles: a list of dimensions of articles
    :returns: the all possible combinations of articles sorted by area
    """
    layouts_all = filter(validate_area, product(*articles))
    layouts = map(add_area, layouts_all)
    sorted_layouts = reversed(sorted(layouts, key=lambda k: k['area']))
    return sorted_layouts


def make_step(parent, dimensions):
    """
    This function will put the next article in the page. We give the list of
    potential articles to put in the page, and try to place them. We will
    return the list of new states of page (one for each succesful placed
    article)

    :params dict parent: the current state of page
    :param list dimensions: the list of articles to put in the page
    :returns: a list with new states of page
    :rtype: list
    """
    ret = []
    for dim in dimensions:
        r = deepcopy(parent)
        coords = r['coords']
        p = r['page']
        row, column = find_empty_cel(p)
        cols = dim['cols']
        rows = dim['rows']
        gap = dim['gap']
        id = dim['id']
        if validate_step(row, column, rows + gap, cols, p):
            p[row:row + rows + gap, column:column + cols] = -1
            coords.append({'id': id, 'col': column, 'row': row, 'cols': cols})
            r['ids'].append(id)
            ret.append(r)
    return ret


def find_solution(layout, rows, columns):
    page = create_matrix(rows, columns)
    solutions = []
    ret = []
    for dim in layout['dimensions']:
        p = page.copy()
        cols = dim['cols']
        rows = dim['rows']
        gap = dim['gap']
        id = dim['id']
        p[0:rows + gap, 0:cols] = -1
        ret.append({'ids': [id],
                    'coords': [{
                        'id': id,
                        'col': 0,
                        'row': 0,
                        'cols': cols
                    }],
                    'page': p})
    while ret:
        dimensions = layout['dimensions']
        variant = ret.pop()
        dims = [dim for dim in dimensions if dim['id'] not in variant['ids']]
        if not dims:
            solutions.append(variant)
        steps = make_step(variant, dims)
        ret = steps + ret
    return solutions


def fast_verify(layout, page_rows, page_columns):
    page = create_matrix(page_rows, page_columns)
    dims = layout['dimensions']
    dimensions = reversed(sorted(dims, key=lambda k: k['area']))
    for dim in dimensions:
        cols = dim['cols']
        rows = dim['rows']
        gap = dim['gap']
        article_area = create_matrix(rows + gap, cols)
        max_peak = np.prod(article_area.shape)
        c = signal.correlate(page, article_area, 'valid')
        rs, cs = np.where(c == max_peak)
        if not list(rs) or not list(cs):
            return False
        row, column = (int(rs[0]), int(cs[0]))
        page[row:row + rows + gap, column:column + cols] = -1
    return True


def generator_fast_verify(layouts):
    for i, layout in enumerate(layouts):
        if fast_verify(layout):
            yield layout


def generate_tile_orders(articles, rows, columns):
    layouts = generate_layouts(articles)
    count = 0
    for layout in generator_fast_verify(layouts, rows, columns):
        sols = find_solution(layout, False)
        if sols and count < 5:
            count += 1
            yield sols
