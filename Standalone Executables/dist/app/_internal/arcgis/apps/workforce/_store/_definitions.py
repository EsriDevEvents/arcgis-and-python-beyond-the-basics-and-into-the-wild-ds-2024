"""
Imports the layer and popup definitions to create services and webmaps for workforce
"""
import json

assignment_layer_definition_v1 = json.loads(
    """
{
  "currentVersion" : 10.3,
  "id" : 0,
  "name" : "Assignments",
  "type" : "Feature Layer",
  "displayField" : "description",
  "description" : "",
  "copyrightText" : "",
  "defaultVisibility" : true,
  "relationships" : [],
  "isDataVersioned" : false,
  "supportsCalculate" : true,
  "supportsAttachmentsByUploadId" : true,
  "supportsRollbackOnFailureParameter" : true,
  "supportsStatistics" : true,
  "supportsAdvancedQueries" : true,
  "supportsValidateSql" : true,
  "supportsCoordinatesQuantization" : true,
  "supportsApplyEditsWithGlobalIds" : true,
  "advancedQueryCapabilities" : {
    "supportsPagination" : true,
    "supportsQueryRelatedPagination" : true,
    "supportsQueryWithDistance" : true,
    "supportsReturningQueryExtent" : true,
    "supportsStatistics" : true,
    "supportsOrderBy" : true,
    "supportsDistinct" : true,
    "supportsQueryWithResultType" : true,
    "supportsSqlExpression" : true,
    "supportsReturningGeometryCentroid" : false
  },
  "useStandardizedQueries" : false,
  "geometryType" : "esriGeometryPoint",
  "minScale" : 0,
  "maxScale" : 0,
  "drawingInfo":{"renderer":{"type":"uniqueValue","field1":"status","uniqueValueInfos":[{"value":"0","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"5bdad7c9-66bb-43a3-8050-41d29c44abeb","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6ODk5NTYzOUJBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzOUFBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+OeyIogAAAr1JREFUeNrEl02IEmEcxv+OK8igqTfxA2RlvXQx9SJdqkvQQToudOiw3lqIoOiwsYdlO1V06uZhO0R7DO9R1CEINS+BuLgXP/AgaH5huGj/Z3gHXgZ0sp2hBx4YX2ee38w778d/HHt7e/QX2mIn2VfZ19jb7LD4r80+Z/9g/2RXC4XChVmgwwTsYt9n32XfZKsmeVP2J/YH9lu+gfm/gNPsFwKoKRqNkt/vJ1VVye12a22z2Yym0ykNBgNqNpvy9biBJwwvbwJ+wH7O9nk8HkokEhQMBsnlcq193Pl8Tt1ul+r1Oo3HYzT9Yh8w/I3xXGcqlTK2HbBfst2RSISy2az2lE6n03Qg4Byfz0exWEzrheFwiG65UywWL3K53Ff5XMVw7T77GAeZTIbS6TQpikKbCtfgWmQIHefz+f1VYIzaIx0aDofpskKGBD9ieNIIxst7zQ7gZCugMlzkBcBguEsG32Pf8Hq9WhdZLWQiGwzB0sCYm7v4EY/HyeFwWA5GJrKFdvmpVYB32Fm0hEIhsktSNlg7AGM+XcHUMZunlxGywQALTEWMZgoEAmS3JEZSEQu+tgzaLYmxDbDW+fraa6ckRkih/ySAO/ouY7ckRkcRm7i2qNstiXEOcBVH/X7fdrDEqAJcYQ9brZa2n9olZIMBFpgAn7G/aR3f6dgGlrLBOlNEnXSKlkajQcvl0nIoMpEtdMoVyVSfTu/ZX0ajEZXLZcvByEQ26zP7nbwt/mY/xPtvt9sEWyUpDyPrkV55ygsIRvchDkqlkiVwZCBL6JCh1VXF3nc2ivFbGAyTyUSrLjfdoxeLBVUqFarVanrTM4a+MqsyUQ322NdRJeIGsKVhgTerNDFlcD7eaa/X08vbx0ao3QX9R/bTVQX91rrByL4tf8JwsGoIX/kJw8CC2cfY2t5jI+DE6o+2PwIMAEzRGFYssovaAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"Unassigned"},{"value":"1","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"8f838b07-f1f9-43f1-b9d4-a9e494fd42f6","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6MTFBQjVGQjZBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6MTFBQjVGQjVBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+r7GfdQAAA6VJREFUeNrEV01IVFEUPjPj6Mwo5YD/IyJDrkKZcfzLJMxNMAsJXLaIVHRhEEE/C1tZripctdFRbBG5ipiFCyUKAtNRG0VaiCCizvgbWuCINkrnu7yRx3SfMy+NPvjgvvvOOd979917znmGlpYWSgIpTBfzMtPNdDIdyr0Qc5EZZH5jzvh8vmiigIYEwmbmbeZN5nWmLUG8CPMj8z3zNT/Ar78R9jCfK4ICtbW1VFRURFlZWWS328Xczs4ObW9v0/LyMo2Njan98QAPWXxaj3AHs5t5MS8vj7xeL7ndbrJarae+7v7+PgWDQRoeHqb19XVM/WB2svirZIQ7mc8wqK6upubmZjKZTKQHR0dHNDAwQBMTE7GpJyzerbYxlZeXq6/vMl9g0NbWRo2NjWQ0Gkkv4OPxeAirNT0tVrrB7/d/53iBExuVPXZtFwbt7e1UVVVFZwViIJaCrtbWVle8MHZvD9MO48rKSjovIJbyEtiNPSxujp1P4BazPj8/n/iGZpCDgwMKhUK0tbVFkUhEzNlsNsrOziaHw0FpaWlSP8RcWVmhtbW1ekVrEN8YZ/Mp81JTUxMVFxdLnTc2NsT3gvDe3h4dHh4KYowHwS7GQ2RkZPx5dAwGMpvNNDs7i0srf+93WOoS5hXMVFRUSEU3NzdpampKiGgB92ADWxlUsaFVAmFs6ws1NTXSc4rlnZubo+Pj44TfEzawhU88EBsa0IKmUdnNmku8urp68j2TAWzhI4NKw2VUEr7YIDIgHeqFlo9KwwnhAowyMzM130AvtHxUGgVG+k+AcBiD3d1dqQGOiF5o+ag0wkaliIuzKANKoF5o+ag0FiE8g9HS0pLUuLCwMGE5jH9b+Mig0piB8Ffmz/HxcVFP44E0WFZWllSVgk1paak0dSI2NKAFTURbYH7BDDKPDDk5OSLzpKena4riHmxgK4MqNrQWUpQ+aYh5Y3R0lOrq6kRujUdubq44DlpFAsubmpqqmdFGRkZil0PcFERi1ekt8044HL7W19cnmgAZsIROp1NQD1gIlQnDT8w36nqM5HoPvVsgEKDJyclzO6+IhZiIzbwf6zzVrQ+6M5QfL8of2hbU2LMAgr29vbHLRyzq1+q58Ghoxhsgjhrscrl0913RaJT6+/uJ66662Xt5WrMHfEaeZ17ljWTBbrRYLCIpoJgnam+xtHjL+fn5WHv7IF70Xzf0H5iP9Tb0Z/6F8WErJ/gZOw3YgQgweN4/bb8FGACBa4t7SuzqvgAAAABJRU5ErkJggg==","contentType":"image/png","width":15,"height":15},"label":"Assigned"},{"value":"2","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"ff54f463-9b9b-493c-9174-63bac7b0e04b","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6ODk5NTYzOUZBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzOUVBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+EdUIugAAA85JREFUeNrEV2lIVGEUPT41U0PLFh2n1bSIKKaCFm0xo4SgaINW6EfbjzYqI8jqR2VBtoAtRPWjBULBLBQijFZzKqOcCokyxjJHHSotS6lU6p7hPXjZe47TGJ3hDPd779573nK/77svoNfc6egAgoQ24XDhKGGc0KqecwmdwlJhmdDxPq+wxVvCAC/CwcLlwjnCqcIwL/mahLeEV4Tn5AKa/0Z4jDBTFfRgYfJ0jIyLR//oGFiienmO1dR9QKW7Fs+cr5Fz+7o+nhewVcQf+yK8VpghjEyw9sOGeYswc1wiIsLC273dhqZGXH1oR1ZeNspd73joszBdxI93RDhduJfG/EkpOLohDcGBQfAFza0tWJ91EJeKbmqHdoh4RnvC64RHaZzavB1zJybDH1y+dxurD+/ThutF/Jg2UHR+rNrdNE5vSfdblGAO5lKxu/e8Gba2wqzeI8IedJ6TNAWdBeZSb6IHNUQ8WJufxFJh8pC+/XFi4zbTJLVfP8Lueg6Huxw1YhOWbj1hi05AonUEYsQ2AnOWvXHiVVVlsqp1NjBs2GDOzT3C+J3LVsAWP+SPwJ/ys1c9x+GSbBRWlOBdgxv13754SPtRzQuUul+hZ2gk+kX0QYD89FAUBV2Du+Dao/schmbmXMjjo04QTuCRWYmTDK/4gasMBx9eRMWnatOnwXP0oa8RdLmplUDh0cKIBZNTDOepu7EOZxz5aPje6PV90oe+jGkL5qYGTWoqajVjVPxQw2TFVc/w5nNNh4uJvowxgk7DpqgLPgZEWwydWUi+wixGpxFH4VhaMVHGFalVry8wi9FpxCr4T6Cwp1Rr64yv0mIyN9uDWYxOo1pRN3G8dRsXEBcHX2EWo9NwUthBq/T1S0PnxL4jMTDS0mFR+iZJjBF0Gg4KP+EUzL1707Of/lEQ4VFYaZuNiJBwr6L0oW+0xBjt1dSgSU0Ks/Y9a1mBvcgw4XjrcKSNW4JB3WNNReO6Wz0+9DWCLje1yoPUPilbmHqy4BIWT0uFEvD7Wsu1l49vsCQ33STkfIzBnRItra04kZ+rDbNlX27SGoEQYaFwMrcwNgGdCTYDbAoE/JvBJlCbx9+FG4X1dLhSfKfTRJlLFa0XbtI6T/0CwureRWPVoQzN2e/Wh7lU7BJRhzbgfqz3LeErEaYU3C+Cs9qF1LHjEaj4tsD9aG7G2qxMHMg5r2/2Dul92goTLL8PwqQXlRVd8+13ER4a5umlQ2Qz99beXr53B2uO7Edx2VOtvU1rK/qvG/obwm2+NvR+f8KI4Bl/vp3+2UfbLwEGANeJh/hAHCeKAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"In Progress"},{"value":"3","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"0686b93d-be21-4626-bd96-94b4c74d8a38","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QTIxNDBBQUVBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6QTIxNDBBQURBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+6nUa3wAAAp5JREFUeNrEl01rE1EUhs+MmTSNzVexYqOJUAkobuIHijsVdK2uCi78ARbEhbiodFHalYgrd25ciF2Ja0EUN2LUmo0iBAomNlZbmjTR5tOm7zvclEtA05gZPfAkk8nMeWaGe+eca+y6eE62ER6QBIfBETAG9qr/FsECeA8+gPTy46fNbgmNLmILXAEXwBng75JvHTwHT8ADXEDjb8THwG0ltCOyPyiDYZ94d1piDXrsfY1KU+o/G1IpVqXwuaSfzwu4Afm7XsRXwSwIDQS8svvgsASjAdlhmX+83V+NDSnly/L906rUynXuWgOTkN/bjngSzHAjHA9K7PgeMUxDeonWRktyb5ekmN16Arcgn9WP6byFibY0fjIq8ROjPUvtu8E5PJc5VMyMXDo/8TsxR+10WxqOBaTfYA5NPg15slPM0XuX44cHOyHV5SpfhA7ILV18GZweCHolhkfkdDAnc9OhXLaYc3OcP0YSw2IYhuNi5mRuFeO4az/FCXCKe0L7AuJWaLnpSlB8FAQ5dbrN036Cuemgi05TjWbxR3zidmiOpKle+OIdslwXa44xiu2JZvk8ros1R9SU/xQU5+0qU226LtMceVMVcan/aLgu1hwLFKftCl6oui7WHGmK50GJJYz11K1gblUm+TFPcQa8sqv2l7JrYi03XRlT9Ulz3LOSKaCKOy9ttVqynFlt/5xDU7Denk6PwMtqqSbZVN5xcS71VWoluxV6AR7qZbEGroFCMVcW4lRo+fA45Xq789RfIBzdU9zIvs47ImcO5lIxBWl6q2j4Dx3Qj00BzvKza4tlqWHeBUeHeq7RdrP3Zkm+fVzRm707nSuEzmA3WOQ3hn+ogrn3r9pbpxr6Z+Bmrw1930sYCO/3s3ZybdG2KcAAzgkQPz13w6cAAAAASUVORK5CYII=","contentType":"image/png","width":15,"height":15},"label":"Completed"},{"value":"4","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"b30b6e18-fe3d-499b-ace5-9ecd4daaf4ca","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6MTFBQjVGQjJBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6MTFBQjVGQjFBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+h4gfbQAAAqFJREFUeNq8lz9oE2EYxr+7XNpryJ/LJkm7VOriEo2LdLGOTo4FBzcdLIjgnyFOpU4iLjoILg5ix+Iuig6C2JpFEItdbEqHQmOapLa5JD7P5TvycWgvae7zgQcul+97f/f9u/c941omIwaQBRfg0/AZeBrOy/8q8Ab8Bf4Kl59Vq25YQCMEHIevwpfhOTgREq8Jv4VX4Bd4gNZxwEX4oQR6mhofF45liYRpChumfnc6oglXXVf8PDhQ+/MB7gC+Ogz4BvwAziRjMXFqYkKcGBsTccM4critbldsHx6K7/v7ot5u89YvuAT402Bb8y/9S/ATQicxwjnH8UYaBvXWBW2mZB/2ZQzGuu44pTDwArzEi3PJpCjCphhe7MO+jCG1BPjCv8DctYseNJUS+d4TjyTGYCypRcALQTB372M4y8Z5rGdUYiw5iCwZgMdV8BX4QgobqdifnsjEmIxNhmR5YJ7Nef44id1riOhlyNhS8xh1guAZ+Dzv5CKc4qCU2GTNEHwWTk8OeGSOK8aWRyxNpil3s8haltAthVEw5Qvfew3qlsKY5lWOV/Z/ACuMnH7aEW+3LT/L6JbC2DJlEvdSm24pjA2Cy7zadV3tYIVRJngNrm0iiTOf6hJjb/YKhRqZBK/DH72JRxLXJSU2WeumrJOWeecHKgcdY+7K2FLLqEia/nF6Bb/fQ7myWq9HDmbMvV4p9A5+qaZFTv5Nrn8F61CJcMoZq9Jb2134ll95xoq27bfZhhvwJa4H82d6xPc3gZ/7M3gX0Nf+DxVMfYK55y8S3sC5Y3U5bM7qYAevNRriW39d7wP6SG0TBFMf4B14ttZu23wAprQEZiA2QHnL9lzTnVbLL29vB6G6C/o38L1hC/qRP2EAfD7Kt5O2j7Y/AgwAIkECnmHeVUQAAAAASUVORK5CYII=","contentType":"image/png","width":15,"height":15},"label":"Declined"},{"value":"5","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"8929e02a-338d-48b0-8bcc-d6b826d6b1a3","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QTIxNDBBQUFBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzQTJBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+Y0l40QAAAr1JREFUeNq0l0toE1EUhk9mEhPTNjGKFuMLKvGBm0ZLRdyo4IO6cVlwIbq1IIKPRV2VdiXiyp0ILsQuRRCqRRRdSGO13fhqtOAjxVLbtNM2bcyj/v8wI5dATWNyD3wwczvzf5n03twznrNtG2QF5QXNYA+IgyawyflbCoyCIfAODN95NJ4vF+gpI/aBM+AUOAyCZfIy4Bl4AO7iA+T+R7wPXHeEdu3eFpD1a7wSqjOlfrVhj80tFMWaL8jEdF4+fF1U7+cHuAz5m0rE50EPCEcaTGnZFZTtUb+s8nn++bi/c0vyZSwrgx8zkp4tcGgGdEJ+ayXiTtDNg51b/XK0JSSGIRVVsSjSP2jJp29Zd+ga5D3qNaWRHa70xP6QHG+tXGqH4h7eywynus+dbOxYTsxZ2+VKd2zxS7XFDEXeBXlzqZiz9yaI8OJaSFW5kxehA3KfKj4NDq0NmXKsNSS1LmYymw7HZYu5Ntt5Eo8FxfDU3GtnMtupdjx1kOIYOMCR2Ga/6Colm64YxXtBiEun3DqtpphNB110Gs5slsaIT3SX4mg2nB98Cdeb2sWKo4niKI/qAoZ2seKI6rctN9PBGA/mF4vaZYpjzHA2cZmZK2gXK45Riod5NJ7OaRcrjmGK3wKLWxj3U13FbGebtOikOAlecST5I6tNrGTTlTScPqnXfv7kgixpeOgiMoeSGfe0F01Bxl1O98GLSSsvjxNWzcVPkDll2RPrObinbov8Hi6A9Mj3rJBalZKXBhfdztOMx+rca35yqYG2z6msRBq8si7srVraN/D3G7wC6UP3RBWzEoDN+BHKp7Humjb6xVPhplXAP7X/9awMvM+ozd4N9ZpSMesl+AUOTs4UApyN3NLC6KVNs3x7O4Lr+xKzkprIue3tpVKp7ob+KbhaaUNf9SsMhLereXfS9tL2R4ABAMvSDbwyRXPsAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"Paused"},{"value":"6","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"7d420f65-891b-4f79-9d1a-511ea42ee64b","width":15,"height":15,"imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDY3IDc5LjE1Nzc0NywgMjAxNS8wMy8zMC0yMzo0MDo0MiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6Nzc0NjQ1REE5OTI0MTFFNUEzNUM5NzQzODc1QTc2Q0MiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6Nzc0NjQ1RDk5OTI0MTFFNUEzNUM5NzQzODc1QTc2Q0MiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo3ZTc0NjJhYy1hMWY4LTQ5Y2YtOGFmNy1iNzdmOWM2OWU5NjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+0lZ3dQAAAvtJREFUeNq8l01IG0EUx182CaiHRKEFY6ERQT30kqbaL6HYQqHgwaIogdCvYCtSFTyUHlo8iDn1IGgihDZBW0PEL+pBKPTz0tpqa/cS8ANES2uEFprkYJGkad/bzsqytia77vbBH2Zn5r3fzuzOzBvDDasVcjATyoE6gjqKKkMdYm1fUKuoj6goig/E4+lsAQ1ZwGbUFdRF1FlUQZZ4W6iXqMeoYXyBlBrwMdQ9BhTspMsFhx0OOGC3Q6HNJtTFYzH4tr4On3ge3o6OSv3pBW4h/IMS8E2UF2UtrqiAC11d4Kirg3yLZc/h/kgmgZ+ZgSd9fbC5vExVCdQdlJ8e8CX2BFPHXiocb2qCq4ODYDSbQYn9TKVgqK0N5iYmxKq7NBAp2CTzaRehLcEgVDc2ghrjjEYw5+VJq3rZ6H07fSSN9Nf2UOF6KKQa+iuTgUednfB6ZETe1NNaWOiQg2ku+1BFBKxqaNAMetrthuo/8YqIgXCzdKrdqFpbZSV4AgFNoZcHBoS2z9EoxJaWahlriGNr00Udz7e3A2cyaQo1cJwQk2Izc+GoCwhcjjpFNc76es2hokliE6ucWpwoy4nm5qzrVC2UjGITg1jE5NjfDHanUzH0YUdHTlDRJAwHxzZ8OFhaqhj6JhzOGSpjlFGvEhD2xmJdoTJGCadmetVAd+1uqA1hN9/c1B0qYWxw7BCHr2truo9UwlglL55K6wsLuk+vhMGTJz0l342NCeepXlCKTQxiEZO8V1CzVLMwPa0LVBabWCscy5OEnOWZ3y8AtYZm0ml46ts5ikcxIdgST4QI6trG4uKZYEsLmPPzNYOShVpb6WSi4itUWJ760Nb5gp2bmkHfT03BfY+Hit9R53C0vDwDoYpuOfRSf79q6PzkpAgl6xahf8u56ENYxbwrtb0tfG/AHEqJpdFvGM/fufFxabLnk+9ccvOypC8xj1liT00NzEYiu5bav5YM9SUfBk2wWN7/mdA/R91WmtDv+wqDwAf7uTvpdmn7LcAAq9+ZUerP/SkAAAAASUVORK5CYII=","contentType":"image/png"},"label":"Canceled"}]},"transparency":0},
  "allowGeometryUpdates" : true,
  "hasAttachments" : true,
  "htmlPopupType" : "esriServerHTMLPopupTypeAsHTMLText",
  "hasM" : false,
  "hasZ" : false,
  "objectIdField" : "OBJECTID",
  "globalIdField" : "GlobalID",
  "typeIdField" : "status",
  "fields" : [
    {
      "name" : "OBJECTID",
      "type" : "esriFieldTypeOID",
      "alias" : "OBJECTID",
      "sqlType" : "sqlTypeOther",
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "description",
      "type" : "esriFieldTypeString",
      "alias" : "Description",
      "sqlType" : "sqlTypeOther",
      "length" : 4000,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "status",
      "type" : "esriFieldTypeInteger",
      "alias" : "Status",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" :
      {
        "type" : "codedValue",
        "name" : "ASSIGN_STATUS",
        "codedValues" : [
          {
            "name" : "Unassigned",
            "code" : 0
          },
          {
            "name" : "Assigned",
            "code" : 1
          },
          {
            "name" : "In Progress",
            "code" : 2
          },
          {
            "name" : "Completed",
            "code" : 3
          },
          {
            "name" : "Declined",
            "code" : 4
          },
          {
            "name" : "Paused",
            "code" : 5
          },
          {
            "name" : "Canceled",
            "code" : 6
          }
        ]
      },
      "defaultValue" : null
    },
    {
      "name" : "notes",
      "type" : "esriFieldTypeString",
      "alias" : "Notes",
      "sqlType" : "sqlTypeOther",
      "length" : 4000,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "priority",
      "type" : "esriFieldTypeInteger",
      "alias" : "Priority",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" :
      {
        "type" : "codedValue",
        "name" : "PRIORITY",
        "codedValues" : [
          {
            "name" : "None",
            "code" : 0
          },
          {
            "name" : "Low",
            "code" : 1
          },
          {
            "name" : "Medium",
            "code" : 2
          },
          {
            "name" : "High",
            "code" : 3
          },
          {
            "name" : "Critical",
            "code" : 4
          }
        ]
      },
      "defaultValue" : null
    },
    {
      "name" : "assignmentType",
      "type" : "esriFieldTypeInteger",
      "alias" : "Assignment Type",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" :
      {
        "type" : "codedValue",
        "name" : "ASSIGN_TYPE",
        "codedValues" : [
        ]
      },
      "defaultValue" : null
    },
    {
      "name" : "workOrderId",
      "type" : "esriFieldTypeString",
      "alias" : "WorkOrder ID",
      "sqlType" : "sqlTypeOther",
      "length" : 255,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "dueDate",
      "type" : "esriFieldTypeDate",
      "alias" : "Due Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "workerId",
      "type" : "esriFieldTypeInteger",
      "alias" : "WorkerID",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "GlobalID",
      "type" : "esriFieldTypeGlobalID",
      "alias" : "GlobalID",
      "sqlType" : "sqlTypeOther",
      "length" : 38,
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "location",
      "type" : "esriFieldTypeString",
      "alias" : "Location",
      "sqlType" : "sqlTypeOther",
      "length" : 255,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "declinedComment",
      "type" : "esriFieldTypeString",
      "alias" : "Declined Comment",
      "sqlType" : "sqlTypeOther",
      "length" : 4000,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "assignedDate",
      "type" : "esriFieldTypeDate",
      "alias" : "Assigned on Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "assignmentRead",
      "type" : "esriFieldTypeInteger",
      "alias" : "Assignment Read",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "inProgressDate",
      "type" : "esriFieldTypeDate",
      "alias" : "In Progress Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "completedDate",
      "type" : "esriFieldTypeDate",
      "alias" : "Completed on Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "declinedDate",
      "type" : "esriFieldTypeDate",
      "alias" : "Declined on Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "pausedDate",
      "type" : "esriFieldTypeDate",
      "alias" : "Paused on Date",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "dispatcherId",
      "type" : "esriFieldTypeInteger",
      "alias" : "DispatcherID",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    }
  ],
  "indexes" : [
    {
      "name" : "FDO_GlobalID",
      "fields" : "GlobalID",
      "isAscending" : true,
      "isUnique" : true,
      "description" : ""
    },
    {
      "name" : "user_256.Assignments_Assignments_Shape_sidx",
      "fields" : "Shape",
      "isAscending" : false,
      "isUnique" : false,
      "description" : "Shape Index"
    },
    {
      "name" : "PK__Assignme__F4B70D856220C85A",
      "fields" : "OBJECTID",
      "isAscending" : true,
      "isUnique" : true,
      "description" : "clustered, unique, primary key"
    },
    {
      "name": "workerIdIndex",
      "fields": "workerId",
      "isAscending": false,
      "isUnique": false,
      "description": "Worker ID index"
    },
    {
      "name": "dispatcherIdIndex",
      "fields": "dispatcherId",
      "isAscending": false,
      "isUnique": false,
      "description": "Dispatcher ID index"
    }
  ],
  "types" : [
    {
      "id" : "0",
      "name" : "Unassigned",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Unassigned",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 0,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    },
    {
      "id" : "1",
      "name" : "Assigned",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Assigned",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 1,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    },
    {
      "id" : "2",
      "name" : "In Progress",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "In Progress",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 2,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    },
    {
      "id" : "3",
      "name" : "Completed",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Completed",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 3,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    },
    {
      "id" : "4",
      "name" : "Declined",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Declined",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 4,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    },
    {
      "id" : "5",
      "name" : "Paused",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Paused",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 5,
              "description" : null,
              "notes" : null,
              "priority" : 0,
              "assignmentType" : null,
              "workOrderId" : null,
              "dueDate" : null,
              "workerId" : null,
              "location" : null,
              "declinedComment" : null,
              "assignedDate" : null,
              "assignmentRead" : null,
              "inProgressDate" : null,
              "completedDate" : null,
              "declinedDate" : null,
              "pausedDate" : null,
              "dispatcherId" : null
            }
          }
        }
      ]
    }
  ],
  "templates" : [],
  "supportedQueryFormats" : "JSON",
  "hasStaticData" : false,
  "maxRecordCount" : 1000,
  "standardMaxRecordCount" : 32000,
  "tileMaxRecordCount" : 8000,
  "maxRecordCountFactor" : 1,
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync",
  "adminLayerInfo" : {
    "geometryField": {
      "name": "Shape"
    }
  }
}
"""
)

assignment_layer_definition_v2 = json.loads(
    """  
{
  "currentVersion" : 10.7, 
  "id" : 0, 
  "name" : "Assignments", 
  "type" : "Feature Layer", 
  "displayField" : "description", 
  "description" : "", 
  "copyrightText" : "", 
  "defaultVisibility" : true, 
  "editFieldsInfo" : {
    "creationDateField" : "CreationDate", 
    "creatorField" : "Creator", 
    "editDateField" : "EditDate", 
    "editorField" : "Editor"
  }, 
  "relationships" : [], 
  "isDataVersioned" : false, 
  "supportsAppend" : true, 
  "supportsCalculate" : true, 
  "supportsASyncCalculate" : true, 
  "supportsTruncate" : false, 
  "supportsAttachmentsByUploadId" : true, 
  "supportsAttachmentsResizing" : true, 
  "supportsRollbackOnFailureParameter" : true, 
  "supportsStatistics" : true, 
  "supportsExceedsLimitStatistics" : true, 
  "supportsAdvancedQueries" : true, 
  "supportsValidateSql" : true, 
  "supportsCoordinatesQuantization" : true, 
  "supportsFieldDescriptionProperty" : true, 
  "supportsQuantizationEditMode" : true, 
  "supportsApplyEditsWithGlobalIds" : true, 
  "supportsReturningQueryGeometry" : true, 
  "advancedQueryCapabilities" : {
    "supportsPagination" : true, 
    "supportsPaginationOnAggregatedQueries" : true, 
    "supportsQueryRelatedPagination" : true, 
    "supportsQueryWithDistance" : true, 
    "supportsReturningQueryExtent" : true, 
    "supportsStatistics" : true, 
    "supportsOrderBy" : true, 
    "supportsDistinct" : true, 
    "supportsQueryWithResultType" : true, 
    "supportsSqlExpression" : true, 
    "supportsAdvancedQueryRelated" : true, 
    "supportsCountDistinct" : true, 
    "supportsPercentileStatistics" : true, 
    "supportsQueryAttachments" : true, 
    "supportsLod" : true, 
    "supportsQueryWithLodSR" : false, 
    "supportedLodTypes" : [
      "geohash"
    ], 
    "supportsReturningGeometryCentroid" : false, 
    "supportsQueryWithDatumTransformation" : true, 
    "supportsHavingClause" : true, 
    "supportsOutFieldSQLExpression" : true, 
    "supportsMaxRecordCountFactor" : true, 
    "supportsTopFeaturesQuery" : true, 
    "supportsDisjointSpatialRel" : true, 
    "supportsQueryWithCacheHint" : true, 
    "supportsQueryAttachmentsWithReturnUrl" : true
  }, 
  "useStandardizedQueries" : true, 
  "geometryType" : "esriGeometryPoint", 
  "minScale" : 0, 
  "maxScale" : 0, 
  "extent" : {
    "xmin" : -16348803.964744022, 
    "ymin" : 2000812.1785607955, 
    "xmax" : -8015003.77241786, 
    "ymax" : 8314629.614010402, 
    "spatialReference" : {
      "wkid" : 102100, 
      "latestWkid" : 3857
    }
  }, 
  "drawingInfo":{"renderer":{"type":"uniqueValue","field1":"status","uniqueValueInfos":[{"value":"0","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"5bdad7c9-66bb-43a3-8050-41d29c44abeb","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6ODk5NTYzOUJBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzOUFBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+OeyIogAAAr1JREFUeNrEl02IEmEcxv+OK8igqTfxA2RlvXQx9SJdqkvQQToudOiw3lqIoOiwsYdlO1V06uZhO0R7DO9R1CEINS+BuLgXP/AgaH5huGj/Z3gHXgZ0sp2hBx4YX2ee38w778d/HHt7e/QX2mIn2VfZ19jb7LD4r80+Z/9g/2RXC4XChVmgwwTsYt9n32XfZKsmeVP2J/YH9lu+gfm/gNPsFwKoKRqNkt/vJ1VVye12a22z2Yym0ykNBgNqNpvy9biBJwwvbwJ+wH7O9nk8HkokEhQMBsnlcq193Pl8Tt1ul+r1Oo3HYzT9Yh8w/I3xXGcqlTK2HbBfst2RSISy2az2lE6n03Qg4Byfz0exWEzrheFwiG65UywWL3K53Ff5XMVw7T77GAeZTIbS6TQpikKbCtfgWmQIHefz+f1VYIzaIx0aDofpskKGBD9ieNIIxst7zQ7gZCugMlzkBcBguEsG32Pf8Hq9WhdZLWQiGwzB0sCYm7v4EY/HyeFwWA5GJrKFdvmpVYB32Fm0hEIhsktSNlg7AGM+XcHUMZunlxGywQALTEWMZgoEAmS3JEZSEQu+tgzaLYmxDbDW+fraa6ckRkih/ySAO/ouY7ckRkcRm7i2qNstiXEOcBVH/X7fdrDEqAJcYQ9brZa2n9olZIMBFpgAn7G/aR3f6dgGlrLBOlNEnXSKlkajQcvl0nIoMpEtdMoVyVSfTu/ZX0ajEZXLZcvByEQ26zP7nbwt/mY/xPtvt9sEWyUpDyPrkV55ygsIRvchDkqlkiVwZCBL6JCh1VXF3nc2ivFbGAyTyUSrLjfdoxeLBVUqFarVanrTM4a+MqsyUQ322NdRJeIGsKVhgTerNDFlcD7eaa/X08vbx0ao3QX9R/bTVQX91rrByL4tf8JwsGoIX/kJw8CC2cfY2t5jI+DE6o+2PwIMAEzRGFYssovaAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"Unassigned"},{"value":"1","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"8f838b07-f1f9-43f1-b9d4-a9e494fd42f6","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6MTFBQjVGQjZBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6MTFBQjVGQjVBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+r7GfdQAAA6VJREFUeNrEV01IVFEUPjPj6Mwo5YD/IyJDrkKZcfzLJMxNMAsJXLaIVHRhEEE/C1tZripctdFRbBG5ipiFCyUKAtNRG0VaiCCizvgbWuCINkrnu7yRx3SfMy+NPvjgvvvOOd979917znmGlpYWSgIpTBfzMtPNdDIdyr0Qc5EZZH5jzvh8vmiigIYEwmbmbeZN5nWmLUG8CPMj8z3zNT/Ar78R9jCfK4ICtbW1VFRURFlZWWS328Xczs4ObW9v0/LyMo2Njan98QAPWXxaj3AHs5t5MS8vj7xeL7ndbrJarae+7v7+PgWDQRoeHqb19XVM/WB2svirZIQ7mc8wqK6upubmZjKZTKQHR0dHNDAwQBMTE7GpJyzerbYxlZeXq6/vMl9g0NbWRo2NjWQ0Gkkv4OPxeAirNT0tVrrB7/d/53iBExuVPXZtFwbt7e1UVVVFZwViIJaCrtbWVle8MHZvD9MO48rKSjovIJbyEtiNPSxujp1P4BazPj8/n/iGZpCDgwMKhUK0tbVFkUhEzNlsNsrOziaHw0FpaWlSP8RcWVmhtbW1ekVrEN8YZ/Mp81JTUxMVFxdLnTc2NsT3gvDe3h4dHh4KYowHwS7GQ2RkZPx5dAwGMpvNNDs7i0srf+93WOoS5hXMVFRUSEU3NzdpampKiGgB92ADWxlUsaFVAmFs6ws1NTXSc4rlnZubo+Pj44TfEzawhU88EBsa0IKmUdnNmku8urp68j2TAWzhI4NKw2VUEr7YIDIgHeqFlo9KwwnhAowyMzM130AvtHxUGgVG+k+AcBiD3d1dqQGOiF5o+ag0wkaliIuzKANKoF5o+ag0FiE8g9HS0pLUuLCwMGE5jH9b+Mig0piB8Ffmz/HxcVFP44E0WFZWllSVgk1paak0dSI2NKAFTURbYH7BDDKPDDk5OSLzpKena4riHmxgK4MqNrQWUpQ+aYh5Y3R0lOrq6kRujUdubq44DlpFAsubmpqqmdFGRkZil0PcFERi1ekt8044HL7W19cnmgAZsIROp1NQD1gIlQnDT8w36nqM5HoPvVsgEKDJyclzO6+IhZiIzbwf6zzVrQ+6M5QfL8of2hbU2LMAgr29vbHLRyzq1+q58Ghoxhsgjhrscrl0913RaJT6+/uJ66662Xt5WrMHfEaeZ17ljWTBbrRYLCIpoJgnam+xtHjL+fn5WHv7IF70Xzf0H5iP9Tb0Z/6F8WErJ/gZOw3YgQgweN4/bb8FGACBa4t7SuzqvgAAAABJRU5ErkJggg==","contentType":"image/png","width":15,"height":15},"label":"Assigned"},{"value":"2","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"ff54f463-9b9b-493c-9174-63bac7b0e04b","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6ODk5NTYzOUZBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzOUVBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+EdUIugAAA85JREFUeNrEV2lIVGEUPT41U0PLFh2n1bSIKKaCFm0xo4SgaINW6EfbjzYqI8jqR2VBtoAtRPWjBULBLBQijFZzKqOcCokyxjJHHSotS6lU6p7hPXjZe47TGJ3hDPd779573nK/77svoNfc6egAgoQ24XDhKGGc0KqecwmdwlJhmdDxPq+wxVvCAC/CwcLlwjnCqcIwL/mahLeEV4Tn5AKa/0Z4jDBTFfRgYfJ0jIyLR//oGFiienmO1dR9QKW7Fs+cr5Fz+7o+nhewVcQf+yK8VpghjEyw9sOGeYswc1wiIsLC273dhqZGXH1oR1ZeNspd73joszBdxI93RDhduJfG/EkpOLohDcGBQfAFza0tWJ91EJeKbmqHdoh4RnvC64RHaZzavB1zJybDH1y+dxurD+/ThutF/Jg2UHR+rNrdNE5vSfdblGAO5lKxu/e8Gba2wqzeI8IedJ6TNAWdBeZSb6IHNUQ8WJufxFJh8pC+/XFi4zbTJLVfP8Lueg6Huxw1YhOWbj1hi05AonUEYsQ2AnOWvXHiVVVlsqp1NjBs2GDOzT3C+J3LVsAWP+SPwJ/ys1c9x+GSbBRWlOBdgxv13754SPtRzQuUul+hZ2gk+kX0QYD89FAUBV2Du+Dao/schmbmXMjjo04QTuCRWYmTDK/4gasMBx9eRMWnatOnwXP0oa8RdLmplUDh0cKIBZNTDOepu7EOZxz5aPje6PV90oe+jGkL5qYGTWoqajVjVPxQw2TFVc/w5nNNh4uJvowxgk7DpqgLPgZEWwydWUi+wixGpxFH4VhaMVHGFalVry8wi9FpxCr4T6Cwp1Rr64yv0mIyN9uDWYxOo1pRN3G8dRsXEBcHX2EWo9NwUthBq/T1S0PnxL4jMTDS0mFR+iZJjBF0Gg4KP+EUzL1707Of/lEQ4VFYaZuNiJBwr6L0oW+0xBjt1dSgSU0Ks/Y9a1mBvcgw4XjrcKSNW4JB3WNNReO6Wz0+9DWCLje1yoPUPilbmHqy4BIWT0uFEvD7Wsu1l49vsCQ33STkfIzBnRItra04kZ+rDbNlX27SGoEQYaFwMrcwNgGdCTYDbAoE/JvBJlCbx9+FG4X1dLhSfKfTRJlLFa0XbtI6T/0CwureRWPVoQzN2e/Wh7lU7BJRhzbgfqz3LeErEaYU3C+Cs9qF1LHjEaj4tsD9aG7G2qxMHMg5r2/2Dul92goTLL8PwqQXlRVd8+13ER4a5umlQ2Qz99beXr53B2uO7Edx2VOtvU1rK/qvG/obwm2+NvR+f8KI4Bl/vp3+2UfbLwEGANeJh/hAHCeKAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"In Progress"},{"value":"3","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"0686b93d-be21-4626-bd96-94b4c74d8a38","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QTIxNDBBQUVBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6QTIxNDBBQURBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+6nUa3wAAAp5JREFUeNrEl01rE1EUhs+MmTSNzVexYqOJUAkobuIHijsVdK2uCi78ARbEhbiodFHalYgrd25ciF2Ja0EUN2LUmo0iBAomNlZbmjTR5tOm7zvclEtA05gZPfAkk8nMeWaGe+eca+y6eE62ER6QBIfBETAG9qr/FsECeA8+gPTy46fNbgmNLmILXAEXwBng75JvHTwHT8ADXEDjb8THwG0ltCOyPyiDYZ94d1piDXrsfY1KU+o/G1IpVqXwuaSfzwu4Afm7XsRXwSwIDQS8svvgsASjAdlhmX+83V+NDSnly/L906rUynXuWgOTkN/bjngSzHAjHA9K7PgeMUxDeonWRktyb5ekmN16Arcgn9WP6byFibY0fjIq8ROjPUvtu8E5PJc5VMyMXDo/8TsxR+10WxqOBaTfYA5NPg15slPM0XuX44cHOyHV5SpfhA7ILV18GZweCHolhkfkdDAnc9OhXLaYc3OcP0YSw2IYhuNi5mRuFeO4az/FCXCKe0L7AuJWaLnpSlB8FAQ5dbrN036Cuemgi05TjWbxR3zidmiOpKle+OIdslwXa44xiu2JZvk8ros1R9SU/xQU5+0qU226LtMceVMVcan/aLgu1hwLFKftCl6oui7WHGmK50GJJYz11K1gblUm+TFPcQa8sqv2l7JrYi03XRlT9Ulz3LOSKaCKOy9ttVqynFlt/5xDU7Denk6PwMtqqSbZVN5xcS71VWoluxV6AR7qZbEGroFCMVcW4lRo+fA45Xq789RfIBzdU9zIvs47ImcO5lIxBWl6q2j4Dx3Qj00BzvKza4tlqWHeBUeHeq7RdrP3Zkm+fVzRm707nSuEzmA3WOQ3hn+ogrn3r9pbpxr6Z+Bmrw1930sYCO/3s3ZybdG2KcAAzgkQPz13w6cAAAAASUVORK5CYII=","contentType":"image/png","width":15,"height":15},"label":"Completed"},{"value":"4","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"b30b6e18-fe3d-499b-ace5-9ecd4daaf4ca","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6MTFBQjVGQjJBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6MTFBQjVGQjFBQjI4MTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+h4gfbQAAAqFJREFUeNq8lz9oE2EYxr+7XNpryJ/LJkm7VOriEo2LdLGOTo4FBzcdLIjgnyFOpU4iLjoILg5ix+Iuig6C2JpFEItdbEqHQmOapLa5JD7P5TvycWgvae7zgQcul+97f/f9u/c941omIwaQBRfg0/AZeBrOy/8q8Ab8Bf4Kl59Vq25YQCMEHIevwpfhOTgREq8Jv4VX4Bd4gNZxwEX4oQR6mhofF45liYRpChumfnc6oglXXVf8PDhQ+/MB7gC+Ogz4BvwAziRjMXFqYkKcGBsTccM4critbldsHx6K7/v7ot5u89YvuAT402Bb8y/9S/ATQicxwjnH8UYaBvXWBW2mZB/2ZQzGuu44pTDwArzEi3PJpCjCphhe7MO+jCG1BPjCv8DctYseNJUS+d4TjyTGYCypRcALQTB372M4y8Z5rGdUYiw5iCwZgMdV8BX4QgobqdifnsjEmIxNhmR5YJ7Nef44id1riOhlyNhS8xh1guAZ+Dzv5CKc4qCU2GTNEHwWTk8OeGSOK8aWRyxNpil3s8haltAthVEw5Qvfew3qlsKY5lWOV/Z/ACuMnH7aEW+3LT/L6JbC2DJlEvdSm24pjA2Cy7zadV3tYIVRJngNrm0iiTOf6hJjb/YKhRqZBK/DH72JRxLXJSU2WeumrJOWeecHKgcdY+7K2FLLqEia/nF6Bb/fQ7myWq9HDmbMvV4p9A5+qaZFTv5Nrn8F61CJcMoZq9Jb2134ll95xoq27bfZhhvwJa4H82d6xPc3gZ/7M3gX0Nf+DxVMfYK55y8S3sC5Y3U5bM7qYAevNRriW39d7wP6SG0TBFMf4B14ttZu23wAprQEZiA2QHnL9lzTnVbLL29vB6G6C/o38L1hC/qRP2EAfD7Kt5O2j7Y/AgwAIkECnmHeVUQAAAAASUVORK5CYII=","contentType":"image/png","width":15,"height":15},"label":"Declined"},{"value":"5","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"8929e02a-338d-48b0-8bcc-d6b826d6b1a3","imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDE0IDc5LjE1Njc5NywgMjAxNC8wOC8yMC0wOTo1MzowMiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6QTIxNDBBQUFBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6ODk5NTYzQTJBQjJCMTFFNDgwMkQ4QzlCNDM4MjFFQjQiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+Y0l40QAAAr1JREFUeNq0l0toE1EUhk9mEhPTNjGKFuMLKvGBm0ZLRdyo4IO6cVlwIbq1IIKPRV2VdiXiyp0ILsQuRRCqRRRdSGO13fhqtOAjxVLbtNM2bcyj/v8wI5dATWNyD3wwczvzf5n03twznrNtG2QF5QXNYA+IgyawyflbCoyCIfAODN95NJ4vF+gpI/aBM+AUOAyCZfIy4Bl4AO7iA+T+R7wPXHeEdu3eFpD1a7wSqjOlfrVhj80tFMWaL8jEdF4+fF1U7+cHuAz5m0rE50EPCEcaTGnZFZTtUb+s8nn++bi/c0vyZSwrgx8zkp4tcGgGdEJ+ayXiTtDNg51b/XK0JSSGIRVVsSjSP2jJp29Zd+ga5D3qNaWRHa70xP6QHG+tXGqH4h7eywynus+dbOxYTsxZ2+VKd2zxS7XFDEXeBXlzqZiz9yaI8OJaSFW5kxehA3KfKj4NDq0NmXKsNSS1LmYymw7HZYu5Ntt5Eo8FxfDU3GtnMtupdjx1kOIYOMCR2Ga/6Colm64YxXtBiEun3DqtpphNB110Gs5slsaIT3SX4mg2nB98Cdeb2sWKo4niKI/qAoZ2seKI6rctN9PBGA/mF4vaZYpjzHA2cZmZK2gXK45Riod5NJ7OaRcrjmGK3wKLWxj3U13FbGebtOikOAlecST5I6tNrGTTlTScPqnXfv7kgixpeOgiMoeSGfe0F01Bxl1O98GLSSsvjxNWzcVPkDll2RPrObinbov8Hi6A9Mj3rJBalZKXBhfdztOMx+rca35yqYG2z6msRBq8si7srVraN/D3G7wC6UP3RBWzEoDN+BHKp7Humjb6xVPhplXAP7X/9awMvM+ozd4N9ZpSMesl+AUOTs4UApyN3NLC6KVNs3x7O4Lr+xKzkprIue3tpVKp7ob+KbhaaUNf9SsMhLereXfS9tL2R4ABAMvSDbwyRXPsAAAAAElFTkSuQmCC","contentType":"image/png","width":15,"height":15},"label":"Paused"},{"value":"6","symbol":{"angle":0,"xoffset":0,"yoffset":0,"type":"esriPMS","url":"7d420f65-891b-4f79-9d1a-511ea42ee64b","width":15,"height":15,"imageData":"iVBORw0KGgoAAAANSUhEUgAAAB4AAAAeCAYAAAA7MK6iAAAAGXRFWHRTb2Z0d2FyZQBBZG9iZSBJbWFnZVJlYWR5ccllPAAAA3hpVFh0WE1MOmNvbS5hZG9iZS54bXAAAAAAADw/eHBhY2tldCBiZWdpbj0i77u/IiBpZD0iVzVNME1wQ2VoaUh6cmVTek5UY3prYzlkIj8+IDx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuNi1jMDY3IDc5LjE1Nzc0NywgMjAxNS8wMy8zMC0yMzo0MDo0MiAgICAgICAgIj4gPHJkZjpSREYgeG1sbnM6cmRmPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5LzAyLzIyLXJkZi1zeW50YXgtbnMjIj4gPHJkZjpEZXNjcmlwdGlvbiByZGY6YWJvdXQ9IiIgeG1sbnM6eG1wTU09Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC9tbS8iIHhtbG5zOnN0UmVmPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvc1R5cGUvUmVzb3VyY2VSZWYjIiB4bWxuczp4bXA9Imh0dHA6Ly9ucy5hZG9iZS5jb20veGFwLzEuMC8iIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0ieG1wLmRpZDo2OWQ2NDJkMi1jZTA0LTRlNWItYWY4My0yYzAzNGRlMTA5YjIiIHhtcE1NOkRvY3VtZW50SUQ9InhtcC5kaWQ6Nzc0NjQ1REE5OTI0MTFFNUEzNUM5NzQzODc1QTc2Q0MiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6Nzc0NjQ1RDk5OTI0MTFFNUEzNUM5NzQzODc1QTc2Q0MiIHhtcDpDcmVhdG9yVG9vbD0iQWRvYmUgUGhvdG9zaG9wIENDIDIwMTQgKE1hY2ludG9zaCkiPiA8eG1wTU06RGVyaXZlZEZyb20gc3RSZWY6aW5zdGFuY2VJRD0ieG1wLmlpZDo3ZTc0NjJhYy1hMWY4LTQ5Y2YtOGFmNy1iNzdmOWM2OWU5NjIiIHN0UmVmOmRvY3VtZW50SUQ9InhtcC5kaWQ6NjlkNjQyZDItY2UwNC00ZTViLWFmODMtMmMwMzRkZTEwOWIyIi8+IDwvcmRmOkRlc2NyaXB0aW9uPiA8L3JkZjpSREY+IDwveDp4bXBtZXRhPiA8P3hwYWNrZXQgZW5kPSJyIj8+0lZ3dQAAAvtJREFUeNq8l01IG0EUx182CaiHRKEFY6ERQT30kqbaL6HYQqHgwaIogdCvYCtSFTyUHlo8iDn1IGgihDZBW0PEL+pBKPTz0tpqa/cS8ANES2uEFprkYJGkad/bzsqytia77vbBH2Zn5r3fzuzOzBvDDasVcjATyoE6gjqKKkMdYm1fUKuoj6goig/E4+lsAQ1ZwGbUFdRF1FlUQZZ4W6iXqMeoYXyBlBrwMdQ9BhTspMsFhx0OOGC3Q6HNJtTFYzH4tr4On3ge3o6OSv3pBW4h/IMS8E2UF2UtrqiAC11d4Kirg3yLZc/h/kgmgZ+ZgSd9fbC5vExVCdQdlJ8e8CX2BFPHXiocb2qCq4ODYDSbQYn9TKVgqK0N5iYmxKq7NBAp2CTzaRehLcEgVDc2ghrjjEYw5+VJq3rZ6H07fSSN9Nf2UOF6KKQa+iuTgUednfB6ZETe1NNaWOiQg2ku+1BFBKxqaNAMetrthuo/8YqIgXCzdKrdqFpbZSV4AgFNoZcHBoS2z9EoxJaWahlriGNr00Udz7e3A2cyaQo1cJwQk2Izc+GoCwhcjjpFNc76es2hokliE6ucWpwoy4nm5qzrVC2UjGITg1jE5NjfDHanUzH0YUdHTlDRJAwHxzZ8OFhaqhj6JhzOGSpjlFGvEhD2xmJdoTJGCadmetVAd+1uqA1hN9/c1B0qYWxw7BCHr2truo9UwlglL55K6wsLuk+vhMGTJz0l342NCeepXlCKTQxiEZO8V1CzVLMwPa0LVBabWCscy5OEnOWZ3y8AtYZm0ml46ts5ikcxIdgST4QI6trG4uKZYEsLmPPzNYOShVpb6WSi4itUWJ760Nb5gp2bmkHfT03BfY+Hit9R53C0vDwDoYpuOfRSf79q6PzkpAgl6xahf8u56ENYxbwrtb0tfG/AHEqJpdFvGM/fufFxabLnk+9ccvOypC8xj1liT00NzEYiu5bav5YM9SUfBk2wWN7/mdA/R91WmtDv+wqDwAf7uTvpdmn7LcAAq9+ZUerP/SkAAAAASUVORK5CYII=","contentType":"image/png"},"label":"Canceled"}]},"transparency":0}, 
  "allowGeometryUpdates" : true, 
  "hasAttachments" : true, 
  
  "attachmentProperties" : [
    {
      "name" : "name", 
      "isEnabled" : true
    }, 
    {
      "name" : "size", 
      "isEnabled" : true
    }, 
    {
      "name" : "contentType", 
      "isEnabled" : true
    }, 
    {
      "name" : "keywords", 
      "isEnabled" : true
    }, 
    {
      "name" : "exifInfo", 
      "isEnabled" : true
    }
  ], 
  "htmlPopupType" : "esriServerHTMLPopupTypeNone", 
  "hasM" : false, 
  "hasZ" : false, 
  "objectIdField" : "OBJECTID", 
  "uniqueIdField" : 
  {
    "name" : "OBJECTID", 
    "isSystemMaintained" : true
  }, 
  "globalIdField" : "GlobalID", 
  "typeIdField" : "status", 
  "fields" : [
    {
      "name" : "OBJECTID", 
      "type" : "esriFieldTypeOID", 
      "alias" : "OBJECTID", 
      "sqlType" : "sqlTypeOther", 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "description", 
      "type" : "esriFieldTypeString", 
      "alias" : "Description", 
      "sqlType" : "sqlTypeOther", 
      "length" : 4000, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "status", 
      "type" : "esriFieldTypeInteger", 
      "alias" : "Status", 
      "sqlType" : "sqlTypeOther", 
      "nullable" : false, 
      "editable" : true, 
      "domain" : 
      {
        "type" : "codedValue", 
        "name" : "ASSIGN_STATUS", 
        "codedValues" : [
          {
            "name" : "Unassigned", 
            "code" : 0
          }, 
          {
            "name" : "Assigned", 
            "code" : 1
          }, 
          {
            "name" : "In Progress", 
            "code" : 2
          }, 
          {
            "name" : "Completed", 
            "code" : 3
          }, 
          {
            "name" : "Declined", 
            "code" : 4
          }, 
          {
            "name" : "Paused", 
            "code" : 5
          }, 
          {
            "name" : "Canceled", 
            "code" : 6
          }
        ]
      }, 
      "defaultValue" : null
    }, 
    {
      "name" : "notes", 
      "type" : "esriFieldTypeString", 
      "alias" : "Notes", 
      "sqlType" : "sqlTypeOther", 
      "length" : 4000, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "priority", 
      "type" : "esriFieldTypeInteger", 
      "alias" : "Priority", 
      "sqlType" : "sqlTypeOther", 
      "nullable" : false, 
      "editable" : true, 
      "domain" : 
      {
        "type" : "codedValue", 
        "name" : "PRIORITY", 
        "codedValues" : [
          {
            "name" : "None", 
            "code" : 0
          }, 
          {
            "name" : "Low", 
            "code" : 1
          }, 
          {
            "name" : "Medium", 
            "code" : 2
          }, 
          {
            "name" : "High", 
            "code" : 3
          }, 
          {
            "name" : "Critical", 
            "code" : 4
          }
        ]
      }, 
      "defaultValue" : null
    }, 
    {
      "name" : "assignmenttype", 
      "type" : "esriFieldTypeGUID", 
      "alias" : "Assignment Type", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "workorderid", 
      "type" : "esriFieldTypeString", 
      "alias" : "WorkOrder ID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 255, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "duedate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "Due Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "workerid", 
      "type" : "esriFieldTypeGUID", 
      "alias" : "WorkerID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "GlobalID", 
      "type" : "esriFieldTypeGlobalID", 
      "alias" : "GlobalID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "location", 
      "type" : "esriFieldTypeString", 
      "alias" : "Location", 
      "sqlType" : "sqlTypeOther", 
      "length" : 255, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "declinedcomment", 
      "type" : "esriFieldTypeString", 
      "alias" : "Declined Comment", 
      "sqlType" : "sqlTypeOther", 
      "length" : 4000, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "assigneddate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "Assigned on Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "inprogressdate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "In Progress Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "completeddate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "Completed on Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "declineddate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "Declined on Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "pauseddate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "Paused on Date", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "dispatcherid", 
      "type" : "esriFieldTypeGUID", 
      "alias" : "DispatcherID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "CreationDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "CreationDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Creator", 
      "type" : "esriFieldTypeString", 
      "alias" : "Creator", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "EditDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "EditDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Editor", 
      "type" : "esriFieldTypeString", 
      "alias" : "Editor", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }
  ], 
  "indexes" : [
    {
      "name" : "PK__workforc__F4B70D855CF94234", 
      "fields" : "OBJECTID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : "clustered, unique, primary key"
    }, 
    {
      "name" : "FDO_GlobalID", 
      "fields" : "GlobalID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : ""
    }, 
    {
      "name" : "workerIdIndex", 
      "fields" : "workerid", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Worker ID index"
    }, 
    {
      "name" : "dispatcherIdIndex", 
      "fields" : "dispatcherid", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Dispatcher ID index"
    }, 
    {
      "name" : "user_51.workforce_a59c183b29d74bf7a202afa6e5cfd1f0_ASSIGNMENTS_Shape_sidx", 
      "fields" : "Shape", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Shape Index"
    }, 
    {
      "name" : "CreationDateIndex", 
      "fields" : "CreationDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "CreationDate Field index"
    }, 
    {
      "name" : "CreatorIndex", 
      "fields" : "Creator", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Creator Field index"
    }, 
    {
      "name" : "EditDateIndex", 
      "fields" : "EditDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "EditDate Field index"
    }, 
    {
      "name" : "EditorIndex", 
      "fields" : "Editor", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Editor Field index"
    }
  ], 
  "types" : [
    {
      "id" : "0", 
      "name" : "Unassigned", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Unassigned", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 0, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "1", 
      "name" : "Assigned", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Assigned", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 1, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "2", 
      "name" : "In Progress", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "In Progress", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 2, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "3", 
      "name" : "Completed", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Completed", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 3, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "4", 
      "name" : "Declined", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Declined", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 4, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "5", 
      "name" : "Paused", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Paused", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 5, 
              "description" : null, 
              "notes" : null, 
              "priority" : 0, 
              "assignmenttype" : null, 
              "workorderid" : null, 
              "duedate" : null, 
              "workerid" : null, 
              "location" : null, 
              "declinedcomment" : null, 
              "assigneddate" : null, 
              "inprogressdate" : null, 
              "completeddate" : null, 
              "declineddate" : null, 
              "pauseddate" : null, 
              "dispatcherid" : null
            }
          }
        }
      ]
    }
  ], 
  "templates" : [], 
  "supportedQueryFormats" : "JSON, geoJSON, PBF", 
  "hasStaticData" : false, 
  "maxRecordCount" : 1000, 
  "standardMaxRecordCount" : 32000, 
  "standardMaxRecordCountNoGeometry" : 32000, 
  "tileMaxRecordCount" : 8000, 
  "maxRecordCountFactor" : 1, 
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync"
}
"""
)

assignment_layer_popup_definition_v1 = json.loads(
    """
{
  "title": "{assignmentType}",
  "fieldInfos": [{
    "fieldName": "OBJECTID",
    "label": "OBJECTID",
    "isEditable": false,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": false
  }, {
    "fieldName": "description",
    "label": "Description",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "status",
    "label": "Status",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "notes",
    "label": "Notes",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "priority",
    "label": "Priority",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "assignmentType",
    "label": "Assignment Type",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "workOrderId",
    "label": "WorkOrder ID",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "dueDate",
    "label": "Due Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "workerId",
    "label": "WorkerID",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "GlobalID",
    "label": "GlobalID",
    "isEditable": false,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": false
  }, {
    "fieldName": "location",
    "label": "Location",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "declinedComment",
    "label": "Declined Comment",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "assignedDate",
    "label": "Assigned on Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "assignmentRead",
    "label": "Assignment Read",
    "isEditable": true,
    "tooltip": "",
    "visible": false,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "inProgressDate",
    "label": "In Progress Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "completedDate",
    "label": "Completed on Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "declinedDate",
    "label": "Declined on Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "pausedDate",
    "label": "Paused on Date",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "dateFormat": "longMonthDayYear"
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "dispatcherId",
    "label": "DispatcherID",
    "isEditable": true,
    "tooltip": "",
    "visible": false,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "CreationDate",
    "label": "CreationDate",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "format": {
      "dateFormat": "shortDateShortTime",
      "timezone": "utc"
    }
  }, {
    "fieldName": "Creator",
    "label": "Creator",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "stringFieldOption": "textbox"
  }, {
    "fieldName": "EditDate",
    "label": "EditDate",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "format": {
      "dateFormat": "shortDateShortTime",
      "timezone": "utc"
    }
  }, {
    "fieldName": "Editor",
    "label": "Editor",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "stringFieldOption": "textbox"
  }],
  "description": null,
  "showAttachments": true,
  "mediaInfos": []
}
"""
)

assignment_layer_popup_definition_v2 = json.loads(
    """
    {
    "title": "{assignmenttype}",
    "fieldInfos": [
        {
            "fieldName": "OBJECTID",
            "label": "OBJECTID",
            "isEditable": false,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": false
        },
        {
            "fieldName": "description",
            "label": "Description",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "status",
            "label": "Status",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "notes",
            "label": "Notes",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "priority",
            "label": "Priority",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "assignmenttype",
            "label": "Assignment Type",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "workorderid",
            "label": "WorkOrder ID",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "duedate",
            "label": "Due Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "workerid",
            "label": "WorkerID",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "GlobalID",
            "label": "GlobalID",
            "isEditable": false,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": false
        },
        {
            "fieldName": "location",
            "label": "Location",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "declinedcomment",
            "label": "Declined Comment",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "assigneddate",
            "label": "Assigned on Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "inprogressdate",
            "label": "In Progress Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "completeddate",
            "label": "Completed on Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "declineddate",
            "label": "Declined on Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "pauseddate",
            "label": "Paused on Date",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "dateFormat": "shortDateLongTime"
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "dispatcherid",
            "label": "DispatcherID",
            "isEditable": true,
            "tooltip": "",
            "visible": false,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "CreationDate",
            "label": "CreationDate",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "format": {
                "dateFormat": "shortDateLongTime",
                "timezone": "utc"
            }
        },
        {
            "fieldName": "Creator",
            "label": "Creator",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "stringFieldOption": "textbox"
        },
        {
            "fieldName": "EditDate",
            "label": "EditDate",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "format": {
                "dateFormat": "shortDateLongTime",
                "timezone": "utc"
            }
        },
        {
            "fieldName": "Editor",
            "label": "Editor",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "stringFieldOption": "textbox"
        }
    ],
    "description": null,
    "showAttachments": true,
    "mediaInfos": []
}
"""
)

dispatcher_layer_definition_v1 = json.loads(
    """
{
  "currentVersion": 10.3,
  "id": 0,
  "name": "Dispatchers",
  "type": "Feature Layer",
  "displayField": "name",
  "description": "Dispatchers",
  "copyrightText": "",
  "defaultVisibility": true,
  "relationships": [],
  "isDataVersioned": false,
  "supportsCalculate": true,
  "supportsAttachmentsByUploadId": true,
  "supportsRollbackOnFailureParameter": true,
  "supportsStatistics": true,
  "supportsAdvancedQueries": true,
  "supportsValidateSql": true,
  "supportsCoordinatesQuantization": true,
  "supportsApplyEditsWithGlobalIds": true,
  "advancedQueryCapabilities": {
    "supportsPagination": true,
    "supportsQueryWithDistance": true,
    "supportsReturningQueryExtent": true,
    "supportsStatistics": true,
    "supportsOrderBy": true,
    "supportsDistinct": true,
    "supportsReturningGeometryCentroid": false
  },
  "useStandardizedQueries": false,
  "geometryType": "esriGeometryPoint",
  "minScale": 0,
  "maxScale": 0,
  "drawingInfo": {
    "renderer": {
      "type": "simple",
      "label": "",
      "description": "",
      "symbol": {
        "angle": 0,
        "xoffset": 0,
        "yoffset": 0,
        "type": "esriPMS",
        "url": "895077ca-7ec0-47e8-b09f-15f2493c083e",
        "imageData": "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76LAAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAHBJREFUGJWFz7ENhDAMheGvyCI3ABJSRqGmvTUQY5AdUt0cVAxAyRpXJIHomrNk2X7+nywHfyJ0/btmxI6EFLrlZsKAQ5RtuB5gwlynsdZsaUA0/Bx/Fb0Bu0O8nXAWvQGp3izOExmsD8AlW7ovVny+mk8Xl6f60psAAAAASUVORK5CYII=",
        "contentType": "image/png",
        "width": 6,
        "height": 6
      }
    },
    "transparency": 0
  },
  "allowGeometryUpdates": true,
  "hasAttachments": false,
  "htmlPopupType": "esriServerHTMLPopupTypeAsHTMLText",
  "hasM": false,
  "hasZ": false,
  "objectIdField": "OBJECTID",
  "globalIdField": "GlobalID",
  "typeIdField": "",
  "fields": [{
    "name": "OBJECTID",
    "type": "esriFieldTypeOID",
    "alias": "OBJECTID",
    "sqlType": "sqlTypeOther",
    "length": 4,
    "nullable": false,
    "editable": false,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "name",
    "type": "esriFieldTypeString",
    "alias": "Name",
    "sqlType": "sqlTypeOther",
    "length": 255,
    "nullable": true,
    "editable": true,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "contactNumber",
    "type": "esriFieldTypeString",
    "alias": "Contact Number",
    "sqlType": "sqlTypeOther",
    "length": 50,
    "nullable": true,
    "editable": true,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "userId",
    "type": "esriFieldTypeString",
    "alias": "Named User",
    "sqlType": "sqlTypeOther",
    "length": 128,
    "nullable": true,
    "editable": true,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "GlobalID",
    "type": "esriFieldTypeGlobalID",
    "alias": "GlobalID",
    "sqlType": "sqlTypeOther",
    "length": 38,
    "nullable": false,
    "editable": false,
    "domain": null,
    "defaultValue": null
  }],
  "indexes": [{
    "name": "PK__DISPATCH__F4B70D859D17E288",
    "fields": "OBJECTID",
    "isAscending": true,
    "isUnique": true,
    "description": "clustered, unique, primary key"
  }, {
    "name": "user_71.DISPATCHERS_DISPATCHERS_Shape_sidx",
    "fields": "Shape",
    "isAscending": false,
    "isUnique": false,
    "description": "Shape Index"
  }, {
    "name": "FDO_GlobalID",
    "fields": "GlobalID",
    "isAscending": true,
    "isUnique": true,
    "description": ""
  }],
  "types": [],
  "templates": [{
    "name": "New Feature",
    "description": "",
    "drawingTool": "esriFeatureEditToolNone",
    "prototype": {
      "attributes": {
        "name": null,
        "contactNumber": null,
        "userId": null
      }
    }
  }],
  "supportedQueryFormats": "JSON",
  "hasStaticData": false,
  "maxRecordCount": 1000,
  "capabilities": "Create,Delete,Query,Update,Editing,Sync"
}
"""
)

dispatcher_table_definition_v2 = json.loads(
    """
{
  "currentVersion" : 10.7, 
  "id" : 2, 
  "name" : "Dispatchers", 
  "type" : "Table", 
  "displayField" : "name", 
  "description" : "", 
  "copyrightText" : "", 
  "defaultVisibility" : true, 
  "editFieldsInfo" : {
    "creationDateField" : "CreationDate", 
    "creatorField" : "Creator", 
    "editDateField" : "EditDate", 
    "editorField" : "Editor"
  }, 
  "relationships" : [], 
  "isDataVersioned" : false, 
  "supportsAppend" : true, 
  "supportsCalculate" : true, 
  "supportsASyncCalculate" : true, 
  "supportsTruncate" : false, 
  "supportsAttachmentsByUploadId" : true, 
  "supportsAttachmentsResizing" : true, 
  "supportsRollbackOnFailureParameter" : true, 
  "supportsStatistics" : true, 
  "supportsExceedsLimitStatistics" : true, 
  "supportsAdvancedQueries" : true, 
  "supportsValidateSql" : true, 
  "supportsCoordinatesQuantization" : true, 
  "supportsFieldDescriptionProperty" : true, 
  "supportsQuantizationEditMode" : true, 
  "supportsApplyEditsWithGlobalIds" : true, 
  "advancedQueryCapabilities" : {
    "supportsPagination" : true, 
    "supportsPaginationOnAggregatedQueries" : true, 
    "supportsQueryRelatedPagination" : true, 
    "supportsQueryWithDistance" : true, 
    "supportsReturningQueryExtent" : true, 
    "supportsStatistics" : true, 
    "supportsOrderBy" : true, 
    "supportsDistinct" : true, 
    "supportsQueryWithResultType" : true, 
    "supportsSqlExpression" : true, 
    "supportsAdvancedQueryRelated" : true, 
    "supportsCountDistinct" : true, 
    "supportsPercentileStatistics" : true, 
    "supportsLod" : true, 
    "supportsQueryWithLodSR" : false, 
    "supportedLodTypes" : [
      "geohash"
    ], 
    "supportsReturningGeometryCentroid" : false, 
    "supportsQueryWithDatumTransformation" : true, 
    "supportsHavingClause" : true, 
    "supportsOutFieldSQLExpression" : true, 
    "supportsMaxRecordCountFactor" : true, 
    "supportsTopFeaturesQuery" : true, 
    "supportsQueryWithCacheHint" : true
  }, 
  "useStandardizedQueries" : true, 
  "allowGeometryUpdates" : true, 
  "hasAttachments" : false, 
  "htmlPopupType" : "esriServerHTMLPopupTypeNone", 
  "hasM" : false, 
  "hasZ" : false, 
  "objectIdField" : "OBJECTID", 
  "uniqueIdField" : 
  {
    "name" : "OBJECTID", 
    "isSystemMaintained" : true
  }, 
  "globalIdField" : "GlobalID", 
  "typeIdField" : "", 
  "fields" : [
    {
      "name" : "OBJECTID", 
      "type" : "esriFieldTypeOID", 
      "alias" : "OBJECTID", 
      "sqlType" : "sqlTypeInteger", 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "name", 
      "type" : "esriFieldTypeString", 
      "alias" : "name", 
      "sqlType" : "sqlTypeVarchar", 
      "length" : 255, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "contactnumber", 
      "type" : "esriFieldTypeString", 
      "alias" : "contactnumber", 
      "sqlType" : "sqlTypeVarchar", 
      "length" : 50, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "userid", 
      "type" : "esriFieldTypeString", 
      "alias" : "userid", 
      "sqlType" : "sqlTypeVarchar", 
      "length" : 128, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "GlobalID", 
      "type" : "esriFieldTypeGlobalID", 
      "alias" : "GlobalID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "wfprivileges", 
      "type" : "esriFieldTypeString", 
      "alias" : "Privileges", 
      "sqlType" : "sqlTypeOther", 
      "length" : 256, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "CreationDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "CreationDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Creator", 
      "type" : "esriFieldTypeString", 
      "alias" : "Creator", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "EditDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "EditDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Editor", 
      "type" : "esriFieldTypeString", 
      "alias" : "Editor", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }
  ], 
  "indexes" : [
    {
      "name" : "PK__workforc__F4B70D85FAE19397", 
      "fields" : "OBJECTID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : "clustered, unique, primary key"
    }, 
    {
      "name" : "UK_workforce_b78e7083b433474cbef3dbd07a11d99d_DISPATCHERS_GlobalID", 
      "fields" : "GlobalID", 
      "isAscending" : false, 
      "isUnique" : true, 
      "description" : "nonclustered, unique"
    }, 
    {
      "name" : "CreationDateIndex", 
      "fields" : "CreationDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "CreationDate Field index"
    }, 
    {
      "name" : "CreatorIndex", 
      "fields" : "Creator", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Creator Field index"
    }, 
    {
      "name" : "EditDateIndex", 
      "fields" : "EditDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "EditDate Field index"
    }, 
    {
      "name" : "EditorIndex", 
      "fields" : "Editor", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Editor Field index"
    }
  ], 
  "types" : [], 
  "templates" : [], 
  "supportedQueryFormats" : "JSON, geoJSON, PBF", 
  "hasStaticData" : false, 
  "maxRecordCount" : 2000, 
  "standardMaxRecordCount" : 32000, 
  "tileMaxRecordCount" : 8000, 
  "maxRecordCountFactor" : 1, 
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync"
}
"""
)

tracking_layer_definition_v1 = json.loads(
    """
{
  "currentVersion": 10.3,
  "id": 0,
  "name": "Location Tracking",
  "type": "Feature Layer",
  "displayField": "OBJECTID",
  "description": "",
  "copyrightText": "",
  "defaultVisibility": true,
  "relationships": [],
  "isDataVersioned": false,
  "supportsCalculate": true,
  "supportsAttachmentsByUploadId": true,
  "supportsRollbackOnFailureParameter": true,
  "supportsStatistics": true,
  "supportsAdvancedQueries": true,
  "supportsValidateSql": true,
  "supportsCoordinatesQuantization": true,
  "supportsApplyEditsWithGlobalIds": true,
  "advancedQueryCapabilities": {
    "supportsPagination": true,
    "supportsQueryWithDistance": true,
    "supportsReturningQueryExtent": true,
    "supportsStatistics": true,
    "supportsOrderBy": true,
    "supportsDistinct": true,
    "supportsReturningGeometryCentroid": false
  },
  "useStandardizedQueries": false,
  "geometryType": "esriGeometryPoint",
  "minScale": 0,
  "maxScale": 0,
  "drawingInfo": {
    "renderer": {
      "type": "simple",
      "symbol": {
        "type": "esriPMS",
        "url": "500274bf-c419-48f7-8ead-99193f3f1d13",
        "imageData": "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAOlJREFUGJV10LFLAnEYxvFvciCCnvKDGxpCkMZoEYIEHdQCB3HVyakU2lpqa2lrEAQRl6ZoaHBx7g9o0EnyGtyEhurF5YiDQx1+Knegz/TA+3nf4TUIJs49OSwivDJmyOdmYGyKelF1J+vcukfuKQfAFT988MYFd4Cj4TNlqUibKOb2dgyLIje8Y1CgqWGa6wDy54waDbrGevtkJwKIYqq8ymi4xNsLl4CHp6EwIsXxTvjNr0ykr+GUByyyJDkMIIdF2A633Ef3T8MqNgMumfNEgnNCRPjnS81URwrSA98fKTMGSv6Dgmz7CpN6Ogm2gMtkAAAAAElFTkSuQmCC",
        "contentType": "image/png",
        "width": 7,
        "height": 7,
        "angle": 0,
        "xoffset": 0,
        "yoffset": 0
      },
      "label": "",
      "description": "Location tracking log for Workforce for ArcGIS"
    },
    "transparency": 0,
    "labelingInfo": null
  },
  "allowGeometryUpdates": true,
  "hasAttachments": false,
  "htmlPopupType": "esriServerHTMLPopupTypeAsHTMLText",
  "hasM": false,
  "hasZ": false,
  "objectIdField": "OBJECTID",
  "globalIdField": "GlobalID",
  "typeIdField": "",
  "fields": [{
    "name": "OBJECTID",
    "type": "esriFieldTypeOID",
    "alias": "OBJECTID",
    "sqlType": "sqlTypeOther",
    "length": 4,
    "nullable": false,
    "editable": false,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "Accuracy",
    "type": "esriFieldTypeDouble",
    "alias": "Accuracy(m)",
    "sqlType": "sqlTypeOther",
    "nullable": true,
    "editable": true,
    "domain": null,
    "defaultValue": null
  }, {
    "name": "GlobalID",
    "type": "esriFieldTypeGlobalID",
    "alias": "",
    "sqlType": "sqlTypeOther",
    "length": 38,
    "nullable": false,
    "editable": false,
    "domain": null,
    "defaultValue": null
  }],
  "indexes": [{
    "name": "GlobalID_Index",
    "fields": "GlobalID",
    "isAscending": false,
    "isUnique": true,
    "description": ""
  }, {
    "name": "user_71.LocationTracking_LOCATION_TRACKING_Shape_sidx",
    "fields": "Shape",
    "isAscending": false,
    "isUnique": false,
    "description": "Shape Index"
  }, {
    "name": "PK__Location__F4B70D8530A46514",
    "fields": "OBJECTID",
    "isAscending": true,
    "isUnique": true,
    "description": "clustered, unique, primary key"
  }],
  "types": [],
  "templates": [{
    "name": "Location Tracking",
    "description": "Location tracking log for Workforce for ArcGIS",
    "drawingTool": "esriFeatureEditToolPoint",
    "prototype": {
      "attributes": {
        "Accuracy": null
      }
    }
  }],
  "supportedQueryFormats": "JSON",
  "hasStaticData": false,
  "maxRecordCount": 1000,
  "capabilities": "Create,Delete,Query,Update,Editing,Sync",
  "adminLayerInfo": {
    "geometryField": {
      "name": "Shape"
    }
  }
}
"""
)

worker_layer_definition_v1 = json.loads(
    """
{
  "currentVersion" : 10.3,
  "id" : 0,
  "name" : "Workers",
  "type" : "Feature Layer",
  "displayField" : "name",
  "description" : "",
  "copyrightText" : "",
  "defaultVisibility" : true,
  "relationships" : [],
  "isDataVersioned" : false,
  "supportsCalculate" : true,
  "supportsAttachmentsByUploadId" : true,
  "supportsRollbackOnFailureParameter" : true,
  "supportsStatistics" : true,
  "supportsAdvancedQueries" : true,
  "supportsValidateSql" : true,
  "supportsCoordinatesQuantization" : true,
  "supportsApplyEditsWithGlobalIds" : true,
  "advancedQueryCapabilities" : {
    "supportsPagination" : true,
    "supportsQueryRelatedPagination" : true,
    "supportsQueryWithDistance" : true,
    "supportsReturningQueryExtent" : true,
    "supportsStatistics" : true,
    "supportsOrderBy" : true,
    "supportsDistinct" : true,
    "supportsQueryWithResultType" : true,
    "supportsSqlExpression" : true,
    "supportsReturningGeometryCentroid" : false
  },
  "useStandardizedQueries" : false,
  "geometryType" : "esriGeometryPoint",
  "minScale" : 0,
  "maxScale" : 0,
  "drawingInfo":{"renderer":{"type":"uniqueValue","field1":"status","field2":null,"field3":null,"fieldDelimiter":", ","uniqueValueInfos":[{"value":"0","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"d18c7af2-ed19-4441-9cec-124931a2ade3","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"Not Working"},{"value":"2","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"036f441c-ee8b-4089-a8e0-2cc35bc8c358","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"On Break"},{"value":"1","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"a6f2c26a-3d18-4155-94cf-68a6c7075de5","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"Working"}]},"transparency":0},
  "allowGeometryUpdates" : true,
  "hasAttachments" : false,
  "htmlPopupType" : "esriServerHTMLPopupTypeAsHTMLText",
  "hasM" : false,
  "hasZ" : false,
  "objectIdField" : "OBJECTID",
  "globalIdField" : "GlobalID",
  "typeIdField" : "status",
  "fields" : [
    {
      "name" : "OBJECTID",
      "type" : "esriFieldTypeOID",
      "alias" : "OBJECTID",
      "sqlType" : "sqlTypeOther",
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "name",
      "type" : "esriFieldTypeString",
      "alias" : "Name",
      "sqlType" : "sqlTypeOther",
      "length" : 255,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "status",
      "type" : "esriFieldTypeInteger",
      "alias" : "Status",
      "sqlType" : "sqlTypeOther",
      "nullable" : true,
      "editable" : true,
      "domain" :
      {
        "type" : "codedValue",
        "name" : "WORKER_STATUS",
        "codedValues" : [
          {
            "name" : "Not Working",
            "code" : 0
          },
          {
            "name" : "Working",
            "code" : 1
          },
          {
            "name" : "On Break",
            "code" : 2
          }
        ]
      },
      "defaultValue" : null
    },
    {
      "name" : "title",
      "type" : "esriFieldTypeString",
      "alias" : "Title",
      "sqlType" : "sqlTypeOther",
      "length" : 255,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "contactNumber",
      "type" : "esriFieldTypeString",
      "alias" : "Contact number",
      "sqlType" : "sqlTypeOther",
      "length" : 50,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "userId",
      "type" : "esriFieldTypeString",
      "alias" : "UserID",
      "sqlType" : "sqlTypeOther",
      "length" : 128,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "notes",
      "type" : "esriFieldTypeString",
      "alias" : "Notes",
      "sqlType" : "sqlTypeOther",
      "length" : 4000,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "GlobalID",
      "type" : "esriFieldTypeGlobalID",
      "alias" : "GlobalID",
      "sqlType" : "sqlTypeOther",
      "length" : 38,
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    }
  ],
  "indexes" : [
    {
      "name" : "FDO_GlobalID",
      "fields" : "GlobalID",
      "isAscending" : true,
      "isUnique" : true,
      "description" : ""
    },
    {
      "name" : "user_256.Workers_Workers_Shape_sidx",
      "fields" : "Shape",
      "isAscending" : false,
      "isUnique" : false,
      "description" : "Shape Index"
    },
    {
      "name" : "PK__Workers___F4B70D8580F1F724",
      "fields" : "OBJECTID",
      "isAscending" : true,
      "isUnique" : true,
      "description" : "clustered, unique, primary key"
    }
  ],
  "types" : [
    {
      "id" : "0",
      "name" : "Not Working",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Not Working",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 0,
              "name" : null,
              "title" : null,
              "contactNumber" : null,
              "userId" : null,
              "notes" : null
            }
          }
        }
      ]
    },
    {
      "id" : "2",
      "name" : "On Break",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "On Break",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 2,
              "name" : null,
              "title" : null,
              "contactNumber" : null,
              "userId" : null,
              "notes" : null
            }
          }
        }
      ]
    },
    {
      "id" : "1",
      "name" : "Working",
      "domains" :
      {
      },
      "templates" : [
        {
          "name" : "Working",
          "description" : "",
          "drawingTool" : "esriFeatureEditToolNone",
          "prototype" : {
            "attributes" : {
              "status" : 1,
              "name" : null,
              "title" : null,
              "contactNumber" : null,
              "userId" : null,
              "notes" : null
            }
          }
        }
      ]
    }
  ],
  "templates" : [],
  "supportedQueryFormats" : "JSON",
  "hasStaticData" : false,
  "maxRecordCount" : 1000,
  "standardMaxRecordCount" : 32000,
  "tileMaxRecordCount" : 8000,
  "maxRecordCountFactor" : 1,
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync",
  "adminLayerInfo" : {
    "geometryField": {
      "name": "Shape"
    }
  }
}
"""
)

worker_layer_definition_v2 = json.loads(
    """
{
  "currentVersion" : 10.7, 
  "id" : 1, 
  "name" : "Workers", 
  "type" : "Feature Layer", 
  "displayField" : "name", 
  "description" : "", 
  "copyrightText" : "", 
  "defaultVisibility" : true, 
  "editFieldsInfo" : {
    "creationDateField" : "CreationDate", 
    "creatorField" : "Creator", 
    "editDateField" : "EditDate", 
    "editorField" : "Editor"
  }, 
  "relationships" : [], 
  "isDataVersioned" : false, 
  "supportsAppend" : true, 
  "supportsCalculate" : true, 
  "supportsASyncCalculate" : true, 
  "supportsTruncate" : false, 
  "supportsAttachmentsByUploadId" : true, 
  "supportsAttachmentsResizing" : true, 
  "supportsRollbackOnFailureParameter" : true, 
  "supportsStatistics" : true, 
  "supportsExceedsLimitStatistics" : true, 
  "supportsAdvancedQueries" : true, 
  "supportsValidateSql" : true, 
  "supportsCoordinatesQuantization" : true, 
  "supportsFieldDescriptionProperty" : true, 
  "supportsQuantizationEditMode" : true, 
  "supportsApplyEditsWithGlobalIds" : true, 
  "supportsReturningQueryGeometry" : true, 
  "advancedQueryCapabilities" : {
    "supportsPagination" : true, 
    "supportsPaginationOnAggregatedQueries" : true, 
    "supportsQueryRelatedPagination" : true, 
    "supportsQueryWithDistance" : true, 
    "supportsReturningQueryExtent" : true, 
    "supportsStatistics" : true, 
    "supportsOrderBy" : true, 
    "supportsDistinct" : true, 
    "supportsQueryWithResultType" : true, 
    "supportsSqlExpression" : true, 
    "supportsAdvancedQueryRelated" : true, 
    "supportsCountDistinct" : true, 
    "supportsPercentileStatistics" : true, 
    "supportsLod" : true, 
    "supportsQueryWithLodSR" : false, 
    "supportedLodTypes" : [
      "geohash"
    ], 
    "supportsReturningGeometryCentroid" : false, 
    "supportsQueryWithDatumTransformation" : true, 
    "supportsHavingClause" : true, 
    "supportsOutFieldSQLExpression" : true, 
    "supportsMaxRecordCountFactor" : true, 
    "supportsTopFeaturesQuery" : true, 
    "supportsDisjointSpatialRel" : true, 
    "supportsQueryWithCacheHint" : true
  }, 
  "useStandardizedQueries" : true, 
  "geometryType" : "esriGeometryPoint", 
  "minScale" : 0, 
  "maxScale" : 0, 
  "extent" : {
    "xmin" : -16348803.964744022, 
    "ymin" : 2000812.1785607955, 
    "xmax" : -8015003.77241786, 
    "ymax" : 8314629.614010402, 
    "spatialReference" : {
      "wkid" : 102100, 
      "latestWkid" : 3857
    }
  }, 
  "drawingInfo":{"renderer":{"type":"uniqueValue","field1":"status","field2":null,"field3":null,"fieldDelimiter":", ","uniqueValueInfos":[{"value":"0","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"d18c7af2-ed19-4441-9cec-124931a2ade3","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"Not Working"},{"value":"2","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"036f441c-ee8b-4089-a8e0-2cc35bc8c358","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"On Break"},{"value":"1","symbol":{"angle":0,"xoffset":0,"yoffset":12,"type":"esriPMS","url":"a6f2c26a-3d18-4155-94cf-68a6c7075de5","width":22.5,"height":22.5,"imageData":"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","contentType":"image/png"},"label":"Working"}]},"transparency":0}, 
  "allowGeometryUpdates" : true, 
  "hasAttachments" : false, 
  "htmlPopupType" : "esriServerHTMLPopupTypeNone", 
  "hasM" : false, 
  "hasZ" : false, 
  "objectIdField" : "OBJECTID", 
  "uniqueIdField" : 
  {
    "name" : "OBJECTID", 
    "isSystemMaintained" : true
  }, 
  "globalIdField" : "GlobalID", 
  "typeIdField" : "status", 
  "fields" : [
    {
      "name" : "OBJECTID", 
      "type" : "esriFieldTypeOID", 
      "alias" : "OBJECTID", 
      "sqlType" : "sqlTypeOther", 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "name", 
      "type" : "esriFieldTypeString", 
      "alias" : "Name", 
      "sqlType" : "sqlTypeOther", 
      "length" : 255, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "status", 
      "type" : "esriFieldTypeInteger", 
      "alias" : "Status", 
      "sqlType" : "sqlTypeOther", 
      "nullable" : false, 
      "editable" : true, 
      "domain" : 
      {
        "type" : "codedValue", 
        "name" : "WORKER_STATUS", 
        "codedValues" : [
          {
            "name" : "Not Working", 
            "code" : 0
          }, 
          {
            "name" : "Working", 
            "code" : 1
          }, 
          {
            "name" : "On Break", 
            "code" : 2
          }
        ]
      }, 
      "defaultValue" : null
    }, 
    {
      "name" : "title", 
      "type" : "esriFieldTypeString", 
      "alias" : "Title", 
      "sqlType" : "sqlTypeOther", 
      "length" : 255, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "contactnumber", 
      "type" : "esriFieldTypeString", 
      "alias" : "Contact number", 
      "sqlType" : "sqlTypeOther", 
      "length" : 50, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "userid", 
      "type" : "esriFieldTypeString", 
      "alias" : "UserID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "notes", 
      "type" : "esriFieldTypeString", 
      "alias" : "Notes", 
      "sqlType" : "sqlTypeOther", 
      "length" : 4000, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "GlobalID", 
      "type" : "esriFieldTypeGlobalID", 
      "alias" : "GlobalID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "wfprivileges", 
      "type" : "esriFieldTypeString", 
      "alias" : "Privileges", 
      "sqlType" : "sqlTypeOther", 
      "length" : 256, 
      "nullable" : true, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "CreationDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "CreationDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Creator", 
      "type" : "esriFieldTypeString", 
      "alias" : "Creator", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "EditDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "EditDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Editor", 
      "type" : "esriFieldTypeString", 
      "alias" : "Editor", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }
  ], 
  "indexes" : [
    {
      "name" : "FDO_GlobalID", 
      "fields" : "GlobalID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : ""
    }, 
    {
      "name" : "user_51.workforce_a59c183b29d74bf7a202afa6e5cfd1f0_WORKERS_Shape_sidx", 
      "fields" : "Shape", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Shape Index"
    }, 
    {
      "name" : "PK__workforc__F4B70D85BFB1DD69", 
      "fields" : "OBJECTID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : "clustered, unique, primary key"
    }, 
    {
      "name" : "CreationDateIndex", 
      "fields" : "CreationDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "CreationDate Field index"
    }, 
    {
      "name" : "CreatorIndex", 
      "fields" : "Creator", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Creator Field index"
    }, 
    {
      "name" : "EditDateIndex", 
      "fields" : "EditDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "EditDate Field index"
    }, 
    {
      "name" : "EditorIndex", 
      "fields" : "Editor", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Editor Field index"
    }
  ], 
  "types" : [
    {
      "id" : "0", 
      "name" : "Not Working", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Not Working", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 0, 
              "name" : null, 
              "title" : null, 
              "contactnumber" : null, 
              "userid" : null, 
              "notes" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "2", 
      "name" : "On Break", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "On Break", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 2, 
              "name" : null, 
              "title" : null, 
              "contactnumber" : null, 
              "userid" : null, 
              "notes" : null
            }
          }
        }
      ]
    }, 
    {
      "id" : "1", 
      "name" : "Working", 
      "domains" : 
      {
      }, 
      "templates" : [
        {
          "name" : "Working", 
          "description" : "", 
          "drawingTool" : "esriFeatureEditToolNone", 
          "prototype" : {
            "attributes" : {
              "status" : 1, 
              "name" : null, 
              "title" : null, 
              "contactnumber" : null, 
              "userid" : null, 
              "notes" : null
            }
          }
        }
      ]
    }
  ], 
  "templates" : [], 
  "supportedQueryFormats" : "JSON, geoJSON, PBF", 
  "hasStaticData" : false, 
  "maxRecordCount" : 1000, 
  "standardMaxRecordCount" : 32000, 
  "standardMaxRecordCountNoGeometry" : 32000, 
  "tileMaxRecordCount" : 8000, 
  "maxRecordCountFactor" : 1, 
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync"
}
"""
)

worker_layer_popup_definition_v1 = json.loads(
    """
{
  "title": "{name}",
  "fieldInfos": [{
    "fieldName": "OBJECTID",
    "label": "OBJECTID",
    "isEditable": false,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": false
  }, {
    "fieldName": "name",
    "label": "Name",
    "isEditable": true,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "status",
    "label": "Status",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "format": {
      "places": 0,
      "digitSeparator": true
    },
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "title",
    "label": "Job Title",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "contactNumber",
    "label": "Contact number",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "userId",
    "label": "UserID",
    "isEditable": true,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "notes",
    "label": "Notes",
    "isEditable": true,
    "tooltip": "",
    "visible": true,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": true
  }, {
    "fieldName": "GlobalID",
    "label": "GlobalID",
    "isEditable": false,
    "tooltip": "",
    "visible": false,
    "stringFieldOption": "textbox",
    "isEditableOnLayer": false
  }, {
    "fieldName": "CreationDate",
    "label": "CreationDate",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "format": {
      "dateFormat": "shortDateShortTime",
      "timezone": "utc"
    }
  }, {
    "fieldName": "Creator",
    "label": "Creator",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "stringFieldOption": "textbox"
  }, {
    "fieldName": "EditDate",
    "label": "EditDate",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "format": {
      "dateFormat": "shortDateShortTime",
      "timezone": "utc"
    }
  }, {
    "fieldName": "Editor",
    "label": "Editor",
    "isEditable": false,
    "isEditableOnLayer": false,
    "visible": false,
    "stringFieldOption": "textbox"
  }],
  "description": null,
  "showAttachments": true,
  "mediaInfos": []
}
"""
)

worker_layer_popup_definition_v2 = json.loads(
    """
    {
    "title": "{name}",
    "fieldInfos": [
        {
            "fieldName": "OBJECTID",
            "label": "OBJECTID",
            "isEditable": false,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": false
        },
        {
            "fieldName": "name",
            "label": "Name",
            "isEditable": true,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "status",
            "label": "Status",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "format": {
                "places": 0,
                "digitSeparator": true
            },
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "title",
            "label": "Job Title",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "contactnumber",
            "label": "Contact number",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "userid",
            "label": "UserID",
            "isEditable": true,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "notes",
            "label": "Notes",
            "isEditable": true,
            "tooltip": "",
            "visible": true,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": true
        },
        {
            "fieldName": "GlobalID",
            "label": "GlobalID",
            "isEditable": false,
            "tooltip": "",
            "visible": false,
            "stringFieldOption": "textbox",
            "isEditableOnLayer": false
        },
        {
            "fieldName": "CreationDate",
            "label": "CreationDate",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "format": {
                "dateFormat": "shortDateLongTime",
                "timezone": "utc"
            }
        },
        {
            "fieldName": "Creator",
            "label": "Creator",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "stringFieldOption": "textbox"
        },
        {
            "fieldName": "EditDate",
            "label": "EditDate",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "format": {
                "dateFormat": "shortDateLongTime",
                "timezone": "utc"
            }
        },
        {
            "fieldName": "Editor",
            "label": "Editor",
            "isEditable": false,
            "isEditableOnLayer": false,
            "visible": false,
            "stringFieldOption": "textbox"
        }
    ],
    "description": null,
    "showAttachments": true,
    "mediaInfos": []
}
"""
)

assignment_type_table_definition_v2 = json.loads(
    """
{
  "currentVersion" : 10.7, 
  "id" : 3, 
  "name" : "Assignment Types", 
  "type" : "Table", 
  "displayField" : "description", 
  "description" : "", 
  "copyrightText" : "", 
  "defaultVisibility" : true, 
  "editFieldsInfo" : {
    "creationDateField" : "CreationDate", 
    "creatorField" : "Creator", 
    "editDateField" : "EditDate", 
    "editorField" : "Editor"
  }, 
  "relationships" : [], 
  "isDataVersioned" : false, 
  "supportsAppend" : true, 
  "supportsCalculate" : true, 
  "supportsASyncCalculate" : true, 
  "supportsTruncate" : false, 
  "supportsAttachmentsByUploadId" : true, 
  "supportsAttachmentsResizing" : true, 
  "supportsRollbackOnFailureParameter" : true, 
  "supportsStatistics" : true, 
  "supportsExceedsLimitStatistics" : true, 
  "supportsAdvancedQueries" : true, 
  "supportsValidateSql" : true, 
  "supportsCoordinatesQuantization" : true, 
  "supportsFieldDescriptionProperty" : true, 
  "supportsQuantizationEditMode" : true, 
  "supportsApplyEditsWithGlobalIds" : true, 
  "advancedQueryCapabilities" : {
    "supportsPagination" : true, 
    "supportsPaginationOnAggregatedQueries" : true, 
    "supportsQueryRelatedPagination" : true, 
    "supportsQueryWithDistance" : true, 
    "supportsReturningQueryExtent" : true, 
    "supportsStatistics" : true, 
    "supportsOrderBy" : true, 
    "supportsDistinct" : true, 
    "supportsQueryWithResultType" : true, 
    "supportsSqlExpression" : true, 
    "supportsAdvancedQueryRelated" : true, 
    "supportsCountDistinct" : true, 
    "supportsPercentileStatistics" : true, 
    "supportsLod" : true, 
    "supportsQueryWithLodSR" : false, 
    "supportedLodTypes" : [
      "geohash"
    ], 
    "supportsReturningGeometryCentroid" : false, 
    "supportsQueryWithDatumTransformation" : true, 
    "supportsHavingClause" : true, 
    "supportsOutFieldSQLExpression" : true, 
    "supportsMaxRecordCountFactor" : true, 
    "supportsTopFeaturesQuery" : true, 
    "supportsQueryWithCacheHint" : true
  }, 
  "useStandardizedQueries" : true, 
  "allowGeometryUpdates" : true, 
  "hasAttachments" : false, 
  "htmlPopupType" : "esriServerHTMLPopupTypeNone", 
  "hasM" : false, 
  "hasZ" : false, 
  "objectIdField" : "OBJECTID", 
  "uniqueIdField" : 
  {
    "name" : "OBJECTID", 
    "isSystemMaintained" : true
  }, 
  "globalIdField" : "GlobalID", 
  "typeIdField" : "", 
  "fields" : [
    {
      "name" : "OBJECTID", 
      "type" : "esriFieldTypeOID", 
      "alias" : "OBJECTID", 
      "sqlType" : "sqlTypeInteger", 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "description", 
      "type" : "esriFieldTypeString", 
      "alias" : "description", 
      "sqlType" : "sqlTypeVarchar", 
      "length" : 255, 
      "nullable" : false, 
      "editable" : true, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "GlobalID", 
      "type" : "esriFieldTypeGlobalID", 
      "alias" : "GlobalID", 
      "sqlType" : "sqlTypeOther", 
      "length" : 38, 
      "nullable" : false, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "CreationDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "CreationDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Creator", 
      "type" : "esriFieldTypeString", 
      "alias" : "Creator", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "EditDate", 
      "type" : "esriFieldTypeDate", 
      "alias" : "EditDate", 
      "sqlType" : "sqlTypeOther", 
      "length" : 8, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }, 
    {
      "name" : "Editor", 
      "type" : "esriFieldTypeString", 
      "alias" : "Editor", 
      "sqlType" : "sqlTypeOther", 
      "length" : 128, 
      "nullable" : true, 
      "editable" : false, 
      "domain" : null, 
      "defaultValue" : null
    }
  ], 
  "indexes" : [
    {
      "name" : "PK__workforc__F4B70D85729BDE38", 
      "fields" : "OBJECTID", 
      "isAscending" : true, 
      "isUnique" : true, 
      "description" : "clustered, unique, primary key"
    }, 
    {
      "name" : "UK_workforce_a59c183b29d74bf7a202afa6e5cfd1f0_ASSIGNMENT_TYPES_GlobalID", 
      "fields" : "GlobalID", 
      "isAscending" : false, 
      "isUnique" : true, 
      "description" : "nonclustered, unique"
    }, 
    {
      "name" : "CreationDateIndex", 
      "fields" : "CreationDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "CreationDate Field index"
    }, 
    {
      "name" : "CreatorIndex", 
      "fields" : "Creator", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Creator Field index"
    }, 
    {
      "name" : "EditDateIndex", 
      "fields" : "EditDate", 
      "isAscending" : true, 
      "isUnique" : false, 
      "description" : "EditDate Field index"
    }, 
    {
      "name" : "EditorIndex", 
      "fields" : "Editor", 
      "isAscending" : false, 
      "isUnique" : false, 
      "description" : "Editor Field index"
    }
  ], 
  "types" : [], 
  "templates" : [], 
  "supportedQueryFormats" : "JSON, geoJSON, PBF", 
  "hasStaticData" : false, 
  "maxRecordCount" : 2000, 
  "standardMaxRecordCount" : 32000, 
  "tileMaxRecordCount" : 8000, 
  "maxRecordCountFactor" : 1, 
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync"
}
"""
)

app_integration_table_definition_v2 = json.loads(
    """
{
  "currentVersion" : 10.7,
  "id" : 4,
  "name" : "Assignment Integrations",
  "type" : "Table",
  "displayField" : "appid",
  "description" : "",
  "copyrightText" : "",
  "defaultVisibility" : true,
  "editFieldsInfo" : {
    "creationDateField" : "CreationDate",
    "creatorField" : "Creator",
    "editDateField" : "EditDate",
    "editorField" : "Editor"
  },
  "relationships" : [],
  "isDataVersioned" : false,
  "supportsAppend" : true,
  "supportsCalculate" : true,
  "supportsASyncCalculate" : true,
  "supportsTruncate" : false,
  "supportsAttachmentsByUploadId" : true,
  "supportsAttachmentsResizing" : true,
  "supportsRollbackOnFailureParameter" : true,
  "supportsStatistics" : true,
  "supportsExceedsLimitStatistics" : true,
  "supportsAdvancedQueries" : true,
  "supportsValidateSql" : true,
  "supportsCoordinatesQuantization" : true,
  "supportsFieldDescriptionProperty" : true,
  "supportsQuantizationEditMode" : true,
  "supportsApplyEditsWithGlobalIds" : true,
  "advancedQueryCapabilities" : {
    "supportsPagination" : true,
    "supportsPaginationOnAggregatedQueries" : true,
    "supportsQueryRelatedPagination" : true,
    "supportsQueryWithDistance" : true,
    "supportsReturningQueryExtent" : true,
    "supportsStatistics" : true,
    "supportsOrderBy" : true,
    "supportsDistinct" : true,
    "supportsQueryWithResultType" : true,
    "supportsSqlExpression" : true,
    "supportsAdvancedQueryRelated" : true,
    "supportsCountDistinct" : true,
    "supportsPercentileStatistics" : true,
    "supportsLod" : true,
    "supportsQueryWithLodSR" : false,
    "supportedLodTypes" : [
      "geohash"
    ],
    "supportsReturningGeometryCentroid" : false,
    "supportsQueryWithDatumTransformation" : true,
    "supportsHavingClause" : true,
    "supportsOutFieldSQLExpression" : true,
    "supportsMaxRecordCountFactor" : true,
    "supportsTopFeaturesQuery" : true,
    "supportsQueryWithCacheHint" : true,
    "supportsQueryAnalytic" : true
  },
  "useStandardizedQueries" : true,
  "allowGeometryUpdates" : true,
  "hasAttachments" : false,
  "htmlPopupType" : "esriServerHTMLPopupTypeNone",
  "hasM" : false,
  "hasZ" : false,
  "objectIdField" : "OBJECTID",
  "uniqueIdField" :
  {
    "name" : "OBJECTID",
    "isSystemMaintained" : true
  },
  "globalIdField" : "GlobalID",
  "typeIdField" : "",
  "fields" : [
    {
      "name" : "OBJECTID",
      "type" : "esriFieldTypeOID",
      "alias" : "OBJECTID",
      "sqlType" : "sqlTypeInteger",
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "GlobalID",
      "type" : "esriFieldTypeGlobalID",
      "alias" : "GlobalID",
      "sqlType" : "sqlTypeOther",
      "length" : 38,
      "nullable" : false,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "appid",
      "type" : "esriFieldTypeString",
      "alias" : "App ID",
      "sqlType" : "sqlTypeVarchar",
      "length" : 255,
      "nullable" : false,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "prompt",
      "type" : "esriFieldTypeString",
      "alias" : "Prompt",
      "sqlType" : "sqlTypeVarchar",
      "length" : 255,
      "nullable" : false,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "urltemplate",
      "type" : "esriFieldTypeString",
      "alias" : "URL Template",
      "sqlType" : "sqlTypeVarchar",
      "length" : 4000,
      "nullable" : false,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "assignmenttype",
      "type" : "esriFieldTypeGUID",
      "alias" : "Assignment Type",
      "sqlType" : "sqlTypeOther",
      "length" : 38,
      "nullable" : true,
      "editable" : true,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "CreationDate",
      "type" : "esriFieldTypeDate",
      "alias" : "CreationDate",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "Creator",
      "type" : "esriFieldTypeString",
      "alias" : "Creator",
      "sqlType" : "sqlTypeOther",
      "length" : 128,
      "nullable" : true,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "EditDate",
      "type" : "esriFieldTypeDate",
      "alias" : "EditDate",
      "sqlType" : "sqlTypeOther",
      "length" : 8,
      "nullable" : true,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    },
    {
      "name" : "Editor",
      "type" : "esriFieldTypeString",
      "alias" : "Editor",
      "sqlType" : "sqlTypeOther",
      "length" : 128,
      "nullable" : true,
      "editable" : false,
      "domain" : null,
      "defaultValue" : null
    }
  ],
  "indexes" : [
    {
      "name" : "PK__workforc__F4B70D858977FB4B",
      "fields" : "OBJECTID",
      "isAscending" : true,
      "isUnique" : true,
      "description" : "clustered, unique, primary key"
    },
    {
      "name" : "UK_workforce_dfedfd13f5784822873688fff9dfba92_ASSIGNMENT_INTEGRATIONS_GlobalID",
      "fields" : "GlobalID",
      "isAscending" : false,
      "isUnique" : true,
      "description" : "nonclustered, unique"
    },
    {
      "name" : "CreationDateIndex",
      "fields" : "CreationDate",
      "isAscending" : true,
      "isUnique" : false,
      "description" : "CreationDate Field index"
    },
    {
      "name" : "CreatorIndex",
      "fields" : "Creator",
      "isAscending" : false,
      "isUnique" : false,
      "description" : "Creator Field index"
    },
    {
      "name" : "EditDateIndex",
      "fields" : "EditDate",
      "isAscending" : true,
      "isUnique" : false,
      "description" : "EditDate Field index"
    },
    {
      "name" : "EditorIndex",
      "fields" : "Editor",
      "isAscending" : false,
      "isUnique" : false,
      "description" : "Editor Field index"
    }
  ],
  "types" : [],
  "templates" : [],
  "supportedQueryFormats" : "JSON, geoJSON, PBF",
  "hasStaticData" : false,
  "maxRecordCount" : 2000,
  "standardMaxRecordCount" : 32000,
  "tileMaxRecordCount" : 8000,
  "maxRecordCountFactor" : 1,
  "capabilities" : "Create,Delete,Query,Update,Editing,Sync"
}
"""
)
